"""Item-based kNN с косинусной мерой и shrinkage.

Сходство треков i и j по столбцам матрицы взаимодействий X:
    s_ij = x_iᵀ x_j / (||x_i|| · ||x_j|| + shrink),
shrink штрафует пары с малым числом общих слушателей. У каждого трека оставляем k ближайших соседей
(по строкам S), диагональ обнуляем.
Оценка трека j для пользователя u — сумма сходств с треками его истории: r_uj = Σ_i x_ui · s_ij.
"""
import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k=100, shrink=10.0, weighting="binary"):
        self.k = k
        self.shrink = shrink
        self.weighting = weighting

    def fit(self, train):
        self.X = train.implicit(self.weighting)
        co = (self.X.T @ self.X).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(co)).copy()
        S = co / (np.outer(norms, norms) + self.shrink + 1e-8)
        np.fill_diagonal(S, 0.0)

        # top-k соседей в каждой строке: s_ij для j из соседей трека i
        k = min(self.k, S.shape[1] - 1)
        cols = np.argpartition(-S, k, axis=1)[:, :k]
        vals = np.take_along_axis(S, cols, axis=1)
        rows = np.repeat(np.arange(S.shape[0]), k)
        self.S = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=S.shape)
        return self

    def score(self, user_idx):
        return (self.X[user_idx] @ self.S).toarray()
