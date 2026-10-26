"""iALS (Hu, Koren, Volinsky, 2008): матричная факторизация для неявной обратной связи.

Для каждой пары (u, i) задаются предпочтение p_ui и уверенность c_ui; для ненаблюдённых пар p = 0, c = 1.
    min_{X, Y}  Σ_{u,i} c_ui (p_ui − x_uᵀ y_i)² + λ (Σ_u ||x_u||² + Σ_i ||y_i||²)
Попеременно решаем точные задачи наименьших квадратов (ALS):
    x_u = (YᵀY + Yᵀ (C_u − I) Y + λI)⁻¹ Yᵀ C_u p_u,
где YᵀY общий для всех пользователей, а поправка считается только по наблюдённым трекам пользователя.
Для треков — симметрично с X.

Базовая версия: p_ui = 1 для наблюдённых пар, c_ui = 1 + α · log(1 + n_ui).
"""
import numpy as np
import polars as pl
import scipy.linalg
import scipy.sparse as sp


def _half_step(conf_m1, target, Y, reg):
    """Решения для всех строк `conf_m1` (CSR, c − 1) при фиксированных факторах противоположной стороны Y."""
    d = Y.shape[1]
    A0 = Y.T @ Y + reg * np.eye(d)
    weighted = conf_m1.data + 1.0  # c_ui
    out = np.zeros((conf_m1.shape[0], d))
    indptr, indices = conf_m1.indptr, conf_m1.indices
    for row in range(conf_m1.shape[0]):
        s, e = indptr[row], indptr[row + 1]
        if s == e:
            continue
        Yr = Y[indices[s:e]]
        A = A0 + (Yr.T * conf_m1.data[s:e]) @ Yr
        b = Yr.T @ (weighted[s:e] * target.data[s:e])
        out[row] = scipy.linalg.solve(A, b, assume_a="pos", check_finite=False)
    return out


def paired_csr(pairs, conf_m1, target, shape):
    """Две CSR-матрицы с одинаковой структурой: c − 1 и p по парам (uidx, iidx); нулевые p сохраняются явно."""
    rows, cols = pairs["uidx"].to_numpy(), pairs["iidx"].to_numpy()
    order = np.lexsort((cols, rows))
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=shape[0]))])
    make = lambda v: sp.csr_matrix((np.asarray(v, dtype=np.float64)[order], cols[order], indptr), shape=shape)
    return make(conf_m1), make(target)


def als(conf_m1, target, factors, reg, iters, seed):
    """Общий ALS для iALS и iALS с кастомным таргетом.

    conf_m1, target — CSR users × items с одинаковым шаблоном: c_ui − 1 и p_ui для наблюдённых пар.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = conf_m1.shape
    X = rng.normal(0, 0.01, (n_users, factors))
    Y = rng.normal(0, 0.01, (n_items, factors))
    conf_t, target_t = conf_m1.T.tocsr(), target.T.tocsr()
    assert conf_m1.nnz == target.nnz and conf_t.nnz == target_t.nnz
    for _ in range(iters):
        X = _half_step(conf_m1, target, Y, reg)
        Y = _half_step(conf_t, target_t, X, reg)
    return X.astype(np.float32), Y.astype(np.float32)


class IALS:
    def __init__(self, factors=64, reg=10.0, alpha=10.0, iters=10, weighting="binary", seed=42):
        self.factors, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.weighting, self.seed = weighting, seed

    def fit(self, train):
        # наблюдённые пары — по той же схеме, что в EASE / ItemKNN: любое прослушивание или только дослушанные
        count = "n_listens" if self.weighting == "binary" else "n_full"
        pairs = train.pairs.filter(pl.col(count) > 0)
        conf_m1, target = paired_csr(pairs, self.alpha * np.log1p(pairs[count].to_numpy()), np.ones(pairs.height),
                                     (train.n_users, train.n_items))
        self.X, self.Y = als(conf_m1, target, self.factors, self.reg, self.iters, self.seed)
        return self

    def score(self, user_idx):
        return self.X[user_idx] @ self.Y.T
