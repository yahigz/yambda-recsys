"""Метрики top-K: NDCG, Recall, Coverage и популярность рекомендаций."""
import numpy as np
import polars as pl


def evaluate(recs, split, k=10):
    """Метрики для рекомендаций `recs` (item_id, форма n_eval_users × k) на сплите `split`.

    - ndcg: бинарная релевантность, IDCG по min(|rel|, k) релевантным на первых позициях;
    - recall: число попаданий / min(|rel|, k) — нормировка как в EASE (Steck, 2019), чтобы пользователи
      с десятками релевантных треков за неделю не занижали метрику до нуля;
    - coverage: доля обучающего каталога, попавшая хотя бы в одну выдачу;
    - mean_pop: средняя доля обучающих пользователей, слушавших рекомендованный трек (чем меньше, тем менее «хитовые» рекомендации).
    """
    recs = recs[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg, recall = np.empty(len(recs)), np.empty(len(recs))
    for i, (uid, row) in enumerate(zip(split.eval_users, recs)):
        rel = split.ground_truth[uid]
        hits = np.fromiter((it in rel for it in row), dtype=bool, count=k)
        n_ideal = min(len(rel), k)
        ndcg[i] = (hits * discounts).sum() / discounts[:n_ideal].sum()
        recall[i] = hits.sum() / n_ideal

    train = split.train
    pop = np.asarray(train.matrix(where=pl.col("n_listens") > 0).sum(axis=0)).ravel() / train.n_users
    rec_idx = np.searchsorted(train.item_ids, recs)
    return {
        f"ndcg@{k}": float(ndcg.mean()),
        f"recall@{k}": float(recall.mean()),
        f"coverage@{k}": len(np.unique(recs)) / train.n_items,
        f"mean_pop@{k}": float(pop[rec_idx].mean()),
    }
