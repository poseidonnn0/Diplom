"""Оценка моделей: бэктест со скользящим началом прогноза, метрики и интервалы."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .dataset import SeriesData
from .models import Forecaster


class HorizonIntervals:
    """Интервал прогноза по методу конформного предсказания.

    Для каждого горизонта берётся квантиль модуля ошибки на отложенных прогнозах.
    Интервал yhat ± q накрывает факт примерно в coverage случаев, какой бы ни была модель.
    """

    def __init__(self, coverage: float = 0.8):
        self.coverage = coverage
        self.q: Optional[np.ndarray] = None

    def fit(self, abs_errors: np.ndarray) -> "HorizonIntervals":
        n = abs_errors.shape[0]
        level = min(1.0, self.coverage * (n + 1) / n)  # поправка на конечную выборку
        q = np.nanquantile(abs_errors, level, axis=0)
        # соседние горизонты ведут себя похоже: сглаживаем, чтобы интервал не «дрожал»
        padded = np.pad(q, 2, mode="edge")
        self.q = np.convolve(padded, np.ones(5) / 5, mode="valid")
        return self

    def apply(self, yhat: np.ndarray):
        q = self.q[: yhat.shape[-1]]
        if len(q) < yhat.shape[-1]:
            q = np.pad(q, (0, yhat.shape[-1] - len(q)), mode="edge")
        return np.maximum(yhat - q, 0.0), yhat + q


@dataclass
class BacktestResult:
    name: str
    metrics: Dict[str, object]
    predictions: np.ndarray = field(repr=False)  # матрица: начало прогноза x горизонт
    actual: np.ndarray = field(repr=False)
    origins: np.ndarray = field(repr=False)

    @property
    def abs_errors(self) -> np.ndarray:
        return np.abs(self.predictions - self.actual)


def horizon_buckets(horizon: int, spd: int) -> Dict[str, slice]:
    """Группы горизонтов для отчёта: до часа, от часа до шести, дальше."""
    hour = max(spd // 24, 1)
    buckets = {"до 1 часа": slice(0, min(hour, horizon))}
    if horizon > hour:
        buckets["1–6 часов"] = slice(hour, min(6 * hour, horizon))
    if horizon > 6 * hour:
        buckets["6–24 часа"] = slice(6 * hour, horizon)
    return buckets


def compute_metrics(predictions: np.ndarray, actual: np.ndarray, spd: int, coverage: float) -> Dict[str, object]:
    errors = predictions - actual
    horizon = predictions.shape[1]
    metrics: Dict[str, object] = {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "wape": float(np.sum(np.abs(errors)) / max(np.sum(np.abs(actual)), 1e-9)),
        "bias": float(np.mean(errors)),
        "origins": int(predictions.shape[0]),
        "by_horizon": {
            label: float(np.mean(np.abs(errors[:, part]))) for label, part in horizon_buckets(horizon, spd).items()
        },
    }
    # Интервалы: границы подбираются на первой половине прогнозов, проверяются на второй.
    half = predictions.shape[0] // 2
    if half >= 5:
        intervals = HorizonIntervals(coverage).fit(np.abs(errors[:half]))
        lo, hi = intervals.apply(predictions[half:])
        inside = (actual[half:] >= lo) & (actual[half:] <= hi)
        metrics["coverage"] = float(np.mean(inside))
        metrics["interval_width"] = float(np.mean(hi - lo))
    return metrics


def backtest(
    model: Forecaster,
    data: SeriesData,
    horizon: int,
    test_steps: int,
    stride: int = 4,
    coverage: float = 0.8,
) -> BacktestResult:
    """Обучить модель на начале ряда и проверить на последних test_steps шагах.

    Прогнозы строятся из каждой stride-й точки тестового участка на horizon шагов вперёд;
    модель при этом видит только прошлое относительно точки начала.
    """
    total = len(data)
    split = total - test_steps
    if split < 2 * data.spd or test_steps <= horizon:
        raise ValueError("Недостаточно данных для бэктеста")
    model.fit(data.head(split))
    origins = np.arange(split - 1, total - horizon, stride)
    predictions = model.predict_many(data, origins, horizon)
    actual = data.y[origins[:, None] + np.arange(1, horizon + 1)[None, :]]
    metrics = compute_metrics(predictions, actual, data.spd, coverage)
    return BacktestResult(model.name, metrics, predictions, actual, origins)
