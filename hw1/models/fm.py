"""Факторизационная машина (Rendle, 2010) с метапризнаками пользователей и треков, обучение BPR.

Пара (u, i) описывается разреженным вектором признаков z = [признаки пользователя, признаки трека] с весами z_f.
    ŷ(z) = w₀ + Σ_f w_f z_f + Σ_{f<g} ⟨v_f, v_g⟩ z_f z_g
         = w₀ + Σ_f w_f z_f + ½ Σ_k [(Σ_f v_fk z_f)² − Σ_f v_fk² z_f²].
Слагаемые, зависящие только от пользователя, одинаковы для всех треков и не влияют на ранжирование и на BPR,
поэтому модель считает только ранжирующую часть:
    s(u, i) = Σ_{f ∈ i} w_f z_f + ⟨Σ_{f ∈ u} v_f z_f, Σ_{g ∈ i} v_g z_g⟩ + ½[(Σ_{g ∈ i} v_g z_g)² − Σ_{g ∈ i} v_g² z_g²].
Скоринг всего каталога для пользователя — одно матричное умножение.

Обучение — BPR: для положительной пары (u, i) и случайного трека j минимизируем −log σ(s(u, i) − s(u, j)) + L2.

Признаки (все считаются по обучающему периоду или статичны — доступны в момент рекомендации).
Набор задаётся `meta`: none — только id; content — id + артист, альбомы, длина; all — всё перечисленное ниже.
- пользователь: id; активность (число треков, квантильный бакет); доля органических прослушиваний; доля дослушанных;
- трек: id; артист; альбомы (multi-hot с весом 1 / число альбомов, не больше 4 самых крупных); длина трека (бакет);
  популярность за весь обучающий период и за последнюю неделю (бакеты).
"""
import numpy as np
import polars as pl
import torch

from .common import DAY

MAX_ALBUMS = 4


def item_metadata(data):
    """Статичные признаки треков: артист, список альбомов, медианная длина. Не зависят от времени."""
    length = data["listens"].group_by("item_id").agg(pl.col("track_length_seconds").median().alias("length"))
    albums = data["album_map"].group_by("item_id").agg(pl.col("album_id").sort().alias("albums"))
    artist = data["artist_map"].group_by("item_id").agg(pl.col("artist_id").min())
    return length.join(albums, on="item_id", how="left").join(artist, on="item_id", how="left")


def _buckets(values, n):
    """Квантильные бакеты 0..n−1."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


class _FeatureSpace:
    """Раскладывает категориальные признаки в общий словарь и строит дополненные массивы индексов и весов."""

    def __init__(self):
        self.size = 1  # индекс 0 — заглушка для дополнения с весом 0

    def block(self, codes):
        codes = np.asarray(codes, dtype=np.int64)
        idx = codes + self.size
        self.size += int(codes.max()) + 1
        return idx


def _pad(columns, weights):
    """Склеивает столбцы признаков (n,) и (n, m) в матрицы индексов и весов одинаковой ширины."""
    idx = np.concatenate([c[:, None] if c.ndim == 1 else c for c in columns], axis=1)
    w = np.concatenate([w[:, None] if w.ndim == 1 else w for w in weights], axis=1)
    return idx, w.astype(np.float32)


class FM:
    def __init__(self, item_meta, dim=64, lr=3e-3, reg=1e-5, epochs=20, n_neg=4, batch_size=4096,
                 meta="all", seed=42, device=None):
        self.item_meta = item_meta
        self.dim, self.lr, self.reg, self.epochs, self.n_neg = dim, lr, reg, epochs, n_neg
        self.batch_size, self.meta, self.seed = batch_size, meta, seed  # meta: none / content / all
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- признаки -------------------------------------------------------------------------------------------
    def _user_features(self, train, space):
        p = train.pairs.filter(pl.col("n_listens") > 0)
        stats = (
            p.group_by("uidx").agg(
                pl.len().alias("n_items"),
                (pl.col("organic_share") * pl.col("n_listens")).sum() / pl.col("n_listens").sum(),
                (pl.col("n_full") > 0).mean().alias("full_share"),
            )
            .rename({"organic_share": "organic"})
        )
        stats = pl.DataFrame({"uidx": np.arange(train.n_users, dtype=np.int32)}).join(stats, on="uidx", how="left").fill_null(0)
        cols = [space.block(np.arange(train.n_users))]
        if self.meta == "all":
            cols += [
                space.block(_buckets(np.log1p(stats["n_items"].to_numpy()), 8)),
                space.block(_buckets(stats["organic"].to_numpy(), 5)),
                space.block(_buckets(stats["full_share"].to_numpy(), 5)),
            ]
        return _pad(cols, [np.ones(train.n_users)] * len(cols))

    def _item_features(self, train, space):
        n = train.n_items
        cols = [space.block(np.arange(n))]
        weights = [np.ones(n)]
        if self.meta in ("content", "all"):
            meta = pl.DataFrame({"item_id": train.item_ids}).join(self.item_meta, on="item_id", how="left")
            listened = train.pairs.filter(pl.col("n_listens") > 0)
            pop = np.bincount(listened["iidx"].to_numpy(), minlength=n)
            recent = listened.filter(pl.col("last_ts") > train.t_end - 7 * DAY)
            pop_recent = np.bincount(recent["iidx"].to_numpy(), minlength=n)

            artist = meta["artist_id"].fill_null(-1).to_numpy()
            _, artist_code = np.unique(artist, return_inverse=True)
            length = meta["length"].fill_null(meta["length"].median()).to_numpy()
            cols += [space.block(artist_code), space.block(_buckets(length, 8))]
            if self.meta == "all":
                cols += [space.block(_buckets(np.log1p(pop), 10)), space.block(_buckets(np.log1p(pop_recent), 5))]
            weights += [np.ones(n)] * (len(cols) - len(weights))

            # альбомы: multi-hot, вес 1 / число альбомов трека; дополняем заглушкой 0 до общей ширины.
            # У сборников бывают сотни альбомов, оставляем MAX_ALBUMS самых крупных в каталоге:
            # альбом с одним треком каталога ничего не обобщает, а ширина матрицы признаков растёт.
            albums = meta["albums"].fill_null([]).to_list()
            size = {}
            for row in albums:
                for a in row:
                    size[a] = size.get(a, 0) + 1
            albums = [sorted(row, key=lambda a: (-size[a], a))[:MAX_ALBUMS] for row in albums]
            vocab = {a: k for k, a in enumerate(sorted({a for row in albums for a in row}))}
            width = max(1, max(len(row) for row in albums))
            alb_idx = np.zeros((n, width), dtype=np.int64)
            alb_w = np.zeros((n, width), dtype=np.float32)
            base = space.size
            for i, row in enumerate(albums):
                for k, a in enumerate(row):
                    alb_idx[i, k] = base + vocab[a]
                    alb_w[i, k] = 1.0 / len(row)
            space.size += len(vocab)
            cols.append(alb_idx)
            weights.append(alb_w)
        return _pad(cols, weights)

    # --- модель ---------------------------------------------------------------------------------------------
    def _side(self, idx, w):
        """Σ v_f z_f и ½[(Σ v z)² − Σ v² z²] для набора объектов."""
        v = self.V(idx) * w.unsqueeze(-1)
        s = v.sum(1)
        return s, 0.5 * (s.pow(2) - v.pow(2).sum(1)).sum(1)

    def _item_terms(self, items):
        idx, w = self.item_idx[items], self.item_w[items]
        s, inner = self._side(idx, w)
        return s, (self.W(idx).squeeze(-1) * w).sum(1) + inner

    def fit(self, train):
        torch.manual_seed(self.seed)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        space = _FeatureSpace()
        u_idx, u_w = self._user_features(train, space)
        i_idx, i_w = self._item_features(train, space)
        dev = self.device
        self.user_idx, self.user_w = torch.as_tensor(u_idx, device=dev), torch.as_tensor(u_w, device=dev)
        self.item_idx, self.item_w = torch.as_tensor(i_idx, device=dev), torch.as_tensor(i_w, device=dev)

        self.V = torch.nn.Embedding(space.size, self.dim, padding_idx=0).to(dev)
        self.W = torch.nn.Embedding(space.size, 1, padding_idx=0).to(dev)
        torch.nn.init.normal_(self.V.weight, std=0.01)
        torch.nn.init.zeros_(self.W.weight)
        with torch.no_grad():
            self.V.weight[0].zero_()
        opt = torch.optim.Adam(list(self.V.parameters()) + list(self.W.parameters()), lr=self.lr)

        # положительные пары: дослушал > 50% или лайкнул, и не дизлайкнул — тот же смысл, что у релевантности
        pos = train.pairs.filter(((pl.col("n_full") > 0) | pl.col("liked")) & ~pl.col("disliked"))
        pu = torch.as_tensor(pos["uidx"].to_numpy(), dtype=torch.long, device=dev)
        pi = torch.as_tensor(pos["iidx"].to_numpy(), dtype=torch.long, device=dev)
        n_items = train.n_items
        for _ in range(self.epochs):
            perm = torch.randperm(len(pu), device=dev, generator=gen)
            for start in range(0, len(pu), self.batch_size):
                b = perm[start:start + self.batch_size]
                u, i = pu[b], pi[b]
                j = torch.randint(0, n_items, (len(b), self.n_neg), device=dev, generator=gen)
                su, _ = self._side(self.user_idx[u], self.user_w[u])
                si, bi = self._item_terms(i)
                sj, bj = self._item_terms(j.reshape(-1))
                sj, bj = sj.view(len(b), self.n_neg, -1), bj.view(len(b), self.n_neg)
                pos_score = bi + (su * si).sum(1)
                neg_score = bj + (su.unsqueeze(1) * sj).sum(2)
                loss = -torch.nn.functional.logsigmoid(pos_score.unsqueeze(1) - neg_score).mean()
                loss = loss + self.reg * (su.pow(2).sum(1).mean() + si.pow(2).sum(1).mean() + sj.pow(2).sum(2).mean())
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            self.user_vec, _ = self._side(self.user_idx, self.user_w)
            self.item_vec, self.item_bias = self._item_terms(torch.arange(n_items, device=dev))
        return self

    def score(self, user_idx):
        with torch.no_grad():
            u = self.user_vec[torch.as_tensor(user_idx, device=self.device, dtype=torch.long)]
            return (self.item_bias.unsqueeze(0) + u @ self.item_vec.T).cpu().numpy()
