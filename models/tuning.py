"""Подбор гиперпараметров на val и финальная оценка на test — одинаково для всех моделей."""
import numpy as np
import optuna

from .common import recommend
from .metrics import evaluate


def run(model_cls, params, split, k=10):
    model = model_cls(**params).fit(split.train)
    return evaluate(recommend(model, split, k), split, k)


def tune(model_cls, space, splits, n_trials, metric="ndcg@10", seed=42, k=10, fixed=None, log_every=0):
    """TPE-подбор на splits["val"] по `metric`, затем обучение на train + val с лучшими параметрами и оценка на test.

    `space(trial)` возвращает словарь параметров, `fixed` — параметры, общие для всех испытаний.
    `log_every` > 0 — печатать прогресс каждые столько испытаний.
    Возвращает (result, trials): result = {"params", "val", "test"}, trials — таблица всех испытаний.
    """
    fixed = fixed or {}

    def objective(trial):
        metrics = run(model_cls, {**space(trial), **fixed}, splits["val"], k)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics[metric]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    def progress(study, trial):
        if log_every and (trial.number + 1) % log_every == 0:
            print(f"  испытание {trial.number + 1}/{n_trials}: {metric} {trial.value:.4f}, "
                  f"лучшее {study.best_value:.4f}, {trial.duration.total_seconds():.0f} с", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    trials = study.trials_dataframe(attrs=("number", "params", "user_attrs", "duration"))
    trials.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in trials.columns]
    trials["duration"] = trials["duration"].dt.total_seconds()
    best = study.best_trial
    params = {**best.params, **fixed}
    result = {"params": best.params, "val": dict(best.user_attrs), "test": run(model_cls, params, splits["test"], k)}
    return result, trials
