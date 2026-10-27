"""Общие функции: загрузка подвыборки, временное разбиение, агрегаты по парам, матрицы и top-k.

Все модели обучаются и оцениваются через одни и те же объекты `Split`, поэтому разбиения, пользователи,
кандидаты и правила исключения у них совпадают.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

DAY = 24 * 3600  # timestamp в Yambda — секунды от начала лога, округлённые до 5 с
LISTEN_PLUS = 50  # played_ratio_pct > 50 — полноценное прослушивание (определение Listen+ авторов Yambda)
FEEDBACK = ("likes", "dislikes", "unlikes", "undislikes")


def load_subsample(path):
    """Читает результат prepare_data.ipynb: события, маппинги и meta.json."""
    import json

    path = Path(path)
    data = {"listens": pl.read_parquet(path / "listens.parquet")}
    for name in FEEDBACK:
        data[name] = pl.read_parquet(path / f"{name}.parquet")
    data["artist_map"] = pl.read_parquet(path / "artist_item_mapping.parquet")
    data["album_map"] = pl.read_parquet(path / "album_item_mapping.parquet")
    data["meta"] = json.loads((path / "meta.json").read_text())
    return data


def _between(t_from, t_to):
    return (pl.col("timestamp") > t_from) & (pl.col("timestamp") <= t_to)


def _last_state(on, off, t_from, t_to, name):
    """Итоговое состояние флага (лайк / дизлайк) по последнему событию «поставил / снял» в окне."""
    events = pl.concat([
        on.select("uid", "item_id", "timestamp").with_columns(pl.lit(True).alias(name)),
        off.select("uid", "item_id", "timestamp").with_columns(pl.lit(False).alias(name)),
    ]).filter(_between(t_from, t_to))
    return events.sort("timestamp").group_by("uid", "item_id").agg(pl.col(name).last())


def aggregate_pairs(data, t_from, t_to):
    """Агрегаты по парам (uid, item_id) за полуинтервал (t_from, t_to].

    Пара попадает в результат, если в окне было хотя бы одно прослушивание или событие лайка / дизлайка.
    """
    listens = data["listens"].filter(_between(t_from, t_to))
    agg = listens.group_by("uid", "item_id").agg(
        pl.len().alias("n_listens"),
        (pl.col("played_ratio_pct") > LISTEN_PLUS).sum().alias("n_full"),
        pl.col("played_ratio_pct").max().alias("max_ratio"),
        pl.col("played_ratio_pct").clip(0, 100).mean().alias("mean_ratio"),
        pl.col("is_organic").mean().alias("organic_share"),
        pl.col("timestamp").min().alias("first_ts"),
        pl.col("timestamp").max().alias("last_ts"),
    )
    liked = _last_state(data["likes"], data["unlikes"], t_from, t_to, "liked")
    disliked = _last_state(data["dislikes"], data["undislikes"], t_from, t_to, "disliked")
    agg = (
        agg.join(liked, on=["uid", "item_id"], how="full", coalesce=True)
        .join(disliked, on=["uid", "item_id"], how="full", coalesce=True)
    )
    return agg.with_columns(
        pl.col("n_listens", "n_full").fill_null(0),
        pl.col("liked", "disliked").fill_null(False),
    )


def is_relevant():
    """Единый критерий релевантности на val / test: дослушал > 50% или лайкнул, и не дизлайкнул."""
    return ((pl.col("n_full") > 0) | pl.col("liked")) & ~pl.col("disliked")


class Interactions:
    """Агрегаты обучающего периода с плотной индексацией пользователей и треков."""

    def __init__(self, pairs, t_end):
        self.t_end = t_end
        self.user_ids = np.sort(pairs["uid"].unique().to_numpy())
        self.item_ids = np.sort(pairs["item_id"].unique().to_numpy())
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)
        uidx = pl.DataFrame({"uid": self.user_ids, "uidx": np.arange(self.n_users, dtype=np.int32)})
        iidx = pl.DataFrame({"item_id": self.item_ids, "iidx": np.arange(self.n_items, dtype=np.int32)})
        self.pairs = pairs.join(uidx, on="uid").join(iidx, on="item_id")

    def matrix(self, values=None, where=None):
        """CSR-матрица users × items. `values` — выражение polars (по умолчанию 1), `where` — фильтр пар."""
        p = self.pairs if where is None else self.pairs.filter(where)
        v = np.ones(p.height, dtype=np.float32) if values is None else p.select(values).to_series().to_numpy()
        return sp.csr_matrix(
            (v.astype(np.float32), (p["uidx"].to_numpy(), p["iidx"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )

    def implicit(self, weighting="binary"):
        """Матрица неявной обратной связи для базовых моделей.

        - binary: 1, если пользователь хоть раз слушал трек;
        - full: 1, если хоть раз дослушал больше чем на 50% (пропуски не считаются интересом);
        - log: log(1 + число прослушиваний) — повторы как сила предпочтения.
        """
        listened = pl.col("n_listens") > 0
        if weighting == "binary":
            return self.matrix(where=listened)
        if weighting == "full":
            return self.matrix(where=pl.col("n_full") > 0)
        if weighting == "log":
            return self.matrix(values=pl.col("n_listens").log1p(), where=listened)
        raise ValueError(weighting)

    def user_index(self, uids):
        return np.searchsorted(self.user_ids, uids)


@dataclass
class Split:
    """Одна постановка «обучаемся на прошлом — оцениваем на следующей неделе»."""

    name: str
    train: Interactions       # всё, что известно модели: пары за (−∞, t_fit]
    eval_users: np.ndarray    # uid пользователей, на которых считаем метрики
    ground_truth: dict        # uid → set(item_id) релевантных новых треков в периоде оценки
    exclude: sp.csr_matrix    # строки eval_users: треки из train, которые не рекомендуем
    t_fit: int
    t_eval: int


def make_split(data, name, t_fit, t_eval):
    """Обучение на (−∞, t_fit], оценка на (t_fit, t_eval].

    Релевантные объекты — новые для пользователя треки (не встречались у него в обучающем периоде),
    удовлетворяющие `is_relevant`. Треки вне обучающего каталога остаются в ground truth: их нельзя
    порекомендовать, и это общий для всех моделей потолок метрик.
    """
    train = Interactions(aggregate_pairs(data, -1, t_fit), t_fit)
    target = (
        aggregate_pairs(data, t_fit, t_eval)
        .filter(is_relevant())
        .join(train.pairs.select("uid", "item_id"), on=["uid", "item_id"], how="anti")
        .filter(pl.col("uid").is_in(pl.Series(train.user_ids).implode()))
    )
    gt = {u: set(items) for u, items in target.group_by("uid").agg(pl.col("item_id")).iter_rows()}
    eval_users = np.array(sorted(gt), dtype=train.user_ids.dtype)
    exclude = train.matrix()[train.user_index(eval_users)]
    return Split(name, train, eval_users, gt, exclude, t_fit, t_eval)


def make_splits(data, val_days=7, test_days=7):
    """Глобальное временное разбиение: train | val | test. Для test обучаемся на train + val."""
    t_max = data["meta"]["t_max"]
    t_test = t_max - test_days * DAY
    t_val = t_test - val_days * DAY
    return {
        "val": make_split(data, "val", t_val, t_test),
        "test": make_split(data, "test", t_test, t_max),
    }


def topk(scores, exclude_rows, k=10):
    """Индексы top-k треков по строкам `scores` (dense, users × items), исключая ненулевые позиции `exclude_rows`."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    rows, cols = exclude_rows.nonzero()
    scores[rows, cols] = -np.inf
    # стабильная сортировка: при равных оценках выигрывает меньший индекс трека, результат не зависит от версии numpy
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def recommend(model, split, k=10, batch_size=1024):
    """Top-k рекомендаций для всех eval-пользователей сплита: массив item_id формы (n_users, k).

    Модель должна уметь `score(user_idx) -> np.ndarray (len(user_idx), n_items)` в индексах `split.train`.
    """
    uidx = split.train.user_index(split.eval_users)
    out = np.empty((len(uidx), k), dtype=split.train.item_ids.dtype)
    for start in range(0, len(uidx), batch_size):
        sl = slice(start, start + batch_size)
        top = topk(model.score(uidx[sl]), split.exclude[sl], k)
        out[sl] = split.train.item_ids[top]
    return out
