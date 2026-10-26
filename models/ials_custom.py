"""iALS с самостоятельно разработанным целевым сигналом: градуированное, выцветающее со временем предпочтение.

Базовый iALS считает любое наблюдённое взаимодействие предпочтением p = 1. В музыке это неверно: пропуск в
первые секунды — скорее сигнал «не нравится», а дизлайк — явный негатив. Строим по каждой паре из обучающего
периода свидетельства «за» и «против»:

    e⁺_ui = n_full_ui + β · liked_ui                       (дослушивания и лайк)
    e⁻_ui = γ · (n_listens_ui − n_full_ui) + D · disliked_ui (пропуски и дизлайк)

Целевой сигнал — доля свидетельств «за», которая выцветает с давностью: интерес к треку, который пользователь
дослушивал два месяца назад, слабее, чем к треку прошлой недели. Уверенность — как в базовом iALS, от объёма свидетельств:

    p_ui = e⁺ / (e⁺ + e⁻) · 2^(−Δt_ui / h)                  ∈ [0, 1]
    c_ui = 1 + α · log(1 + e⁺ + e⁻)

где Δt_ui — дни от последнего события пары до конца обучающего периода, h — период полураспада интереса.
Пара «только пропуски» или с дизлайком получает p ≈ 0 с уверенностью выше, чем у ненаблюдённой пары, —
модель учится отодвигать такие треки. Обучение — тот же ALS, что в `ials.py`.

`decay_in="confidence"` переносит затухание из таргета в уверенность — только для ablation.
"""
import polars as pl

from .common import DAY
from .ials import als, paired_csr


class IALSCustom:
    def __init__(self, factors=64, reg=10.0, alpha=10.0, iters=10, beta=2.0, gamma=0.5, half_life=14.0,
                 dislike_weight=5.0, decay_in="target", seed=42):
        self.factors, self.reg, self.alpha, self.iters = factors, reg, alpha, iters
        self.beta, self.gamma, self.half_life, self.dislike_weight = beta, gamma, half_life, dislike_weight
        self.decay_in, self.seed = decay_in, seed

    def signal(self, train):
        """Пары обучающего периода со столбцами `target` (p_ui) и `conf_m1` (c_ui − 1)."""
        pos = pl.col("n_full") + self.beta * pl.col("liked").cast(pl.Float64)
        neg = (self.gamma * (pl.col("n_listens") - pl.col("n_full"))
               + self.dislike_weight * pl.col("disliked").cast(pl.Float64))
        age_days = (train.t_end - pl.col("last_ts").fill_null(train.t_end)) / DAY
        decay = 2.0 ** (-pl.col("age") / self.half_life)
        in_target = self.decay_in == "target"
        return (
            train.pairs.with_columns(pos.alias("pos"), neg.alias("neg"), age_days.alias("age"))
            .filter(pl.col("pos") + pl.col("neg") > 0)
            .with_columns(
                (pl.col("pos") / (pl.col("pos") + pl.col("neg")) * (decay if in_target else 1.0)).alias("target"),
                (self.alpha * (1 + pl.col("pos") + pl.col("neg")).log() * (1.0 if in_target else decay)).alias("conf_m1"),
            )
        )

    def fit(self, train):
        pairs = self.signal(train)
        conf_m1, target = paired_csr(pairs, pairs["conf_m1"], pairs["target"], (train.n_users, train.n_items))
        self.X, self.Y = als(conf_m1, target, self.factors, self.reg, self.iters, self.seed)
        return self

    def score(self, user_idx):
        return self.X[user_idx] @ self.Y.T
