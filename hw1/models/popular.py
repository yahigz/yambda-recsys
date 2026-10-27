"""Бейзлайн Popular: одинаковый для всех рейтинг треков по числу слушателей за последние `days` дней обучения."""
import numpy as np
import polars as pl

from .common import DAY


class Popular:
    def __init__(self, days=None):
        self.days = days  # None — вся обучающая история

    def fit(self, train):
        where = pl.col("n_listens") > 0
        if self.days is not None:
            where &= pl.col("last_ts") > train.t_end - self.days * DAY
        self.pop = np.asarray(train.matrix(where=where).sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, user_idx):
        return np.broadcast_to(self.pop, (len(user_idx), len(self.pop)))
