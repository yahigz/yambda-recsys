"""EASE (Steck, 2019): линейная item-item модель с замкнутым решением.

Задача: min_B ||X − XB||² + λ||B||²  при diag(B) = 0.
Решение: P = (XᵀX + λI)⁻¹,  B = I − P · diag(1 / diag(P)),  то есть B_ij = −P_ij / P_jj, B_jj = 0.
Предсказание: оценки пользователя u по всем трекам — x_u B.
"""
import numpy as np
import scipy.linalg


class EASE:
    def __init__(self, lam=500.0, weighting="binary"):
        self.lam = lam
        self.weighting = weighting

    def fit(self, train):
        self.X = train.implicit(self.weighting)
        G = (self.X.T @ self.X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, user_idx):
        return np.asarray(self.X[user_idx] @ self.B)
