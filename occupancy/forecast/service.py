"""Обучение, выбор лучшей модели и расчёт прогнозов для интерфейса."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import session_scope
from ..models import Forecast, ModelRun, Zone
from ..store import load_counts
from ..timeutil import floor_to, utcnow
from .dataset import SeriesData, prepare
from .evaluate import BacktestResult, HorizonIntervals, backtest
from .models import candidates, model_titles

log = logging.getLogger("occupancy.forecast")

MIN_DAYS = 10  # меньше этого обучать бессмысленно: нет даже одной полной недели с запасом
KEEP_HORIZONS = (1, 4, 12, 24, 48, 96)  # прошлые прогнозы на эти горизонты остаются в базе для сверки с фактом
LOOKBACK_DAYS = 9  # сколько истории нужно, чтобы построить прогноз

_artifact_cache: Dict[str, tuple] = {}


def zone_series(session: Session, zone_id: int, settings: Settings, end: Optional[datetime] = None,
                lookback_days: Optional[int] = None) -> SeriesData:
    start = end - timedelta(days=lookback_days) if end is not None and lookback_days else None
    counts = load_counts(session, zone_id, start=start, end=end)
    return prepare(counts, settings.freq_minutes, settings.timezone, pd.Timestamp(end) if end is not None else None)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarize(results: Dict[str, BacktestResult]) -> Dict[str, object]:
    """Свести метрики одной модели по зонам в общую строку отчёта."""
    per_zone = {zone: result.metrics for zone, result in results.items()}
    labels = list(next(iter(per_zone.values()))["by_horizon"].keys())
    summary: Dict[str, object] = {
        "mae": _mean(m["mae"] for m in per_zone.values()),
        "rmse": _mean(m["rmse"] for m in per_zone.values()),
        "wape": _mean(m["wape"] for m in per_zone.values()),
        "bias": _mean(m["bias"] for m in per_zone.values()),
        "by_horizon": {label: _mean(m["by_horizon"][label] for m in per_zone.values()) for label in labels},
        "origins": int(sum(m["origins"] for m in per_zone.values())),
        "zones": {zone: {"mae": m["mae"], "rmse": m["rmse"]} for zone, m in per_zone.items()},
    }
    covered = [m["coverage"] for m in per_zone.values() if "coverage" in m]
    if covered:
        summary["coverage"] = _mean(covered)
        summary["interval_width"] = _mean(m["interval_width"] for m in per_zone.values() if "interval_width" in m)
    return summary


def train(
    settings: Optional[Settings] = None,
    include_neural: bool = True,
    only: Optional[List[str]] = None,
    report: Callable[[str], None] = log.info,
) -> dict:
    """Сравнить модели на бэктесте, выбрать лучшую по MAE, обучить её на всех данных и сохранить."""
    settings = settings or get_settings()
    horizon, spd = settings.horizon_steps, settings.steps_per_day
    registry = candidates(horizon, include_neural)
    if only:
        registry = {name: factory for name, factory in registry.items() if name in only}
    if not registry:
        raise ValueError("Не выбрано ни одной модели")

    with session_scope() as session:
        zones = session.scalars(select(Zone).where(Zone.active.is_(True)).order_by(Zone.position, Zone.id)).all()
        series = {}
        for zone in zones:
            data = zone_series(session, zone.id, settings)
            if len(data) < MIN_DAYS * spd:
                report(f"Зона «{zone.name}»: данных {len(data) / spd:.1f} сут., нужно не меньше {MIN_DAYS}. Пропущена.")
                continue
            series[zone.id] = (zone.name, data)
    if not series:
        raise ValueError(f"Нет зон с историей хотя бы за {MIN_DAYS} суток. Соберите данные или загрузите демо-историю.")

    results: Dict[str, Dict[int, BacktestResult]] = {name: {} for name in registry}
    for zone_id, (zone_name, data) in series.items():
        test_steps = max(min(settings.backtest_days * spd, len(data) // 3), horizon + spd)
        for name, factory in registry.items():
            result = backtest(factory(), data, horizon, test_steps, coverage=settings.interval_coverage)
            results[name][zone_id] = result
            report(f"{zone_name}: {model_titles()[name]} — MAE {result.metrics['mae']:.3f}")

    zone_names = {zone_id: name for zone_id, (name, _) in series.items()}
    summary = {
        name: summarize({zone_names[zone_id]: result for zone_id, result in by_zone.items()})
        for name, by_zone in results.items()
    }
    baseline = summary.get("seasonal_naive", {}).get("mae")
    for metrics in summary.values():
        if baseline:
            metrics["skill"] = 1.0 - metrics["mae"] / baseline  # доля ошибки, снятая относительно наивной модели
    best = min(summary, key=lambda name: summary[name]["mae"])
    report(f"Лучшая модель: {model_titles()[best]}")

    packs = {}
    for zone_id, (_, data) in series.items():
        model = registry[best]().fit(data)
        intervals = HorizonIntervals(settings.interval_coverage).fit(results[best][zone_id].abs_errors)
        packs[zone_id] = {"model": model, "intervals": intervals}

    created = utcnow()
    batch = created.strftime("%Y%m%d-%H%M%S")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = settings.artifacts_dir / f"{batch}-{best}.joblib"
    joblib.dump(
        {"name": best, "freq_minutes": settings.freq_minutes, "horizon_steps": horizon,
         "trained_at": created, "zones": packs},
        artifact_path,
    )

    with session_scope() as session:
        session.execute(update(ModelRun).values(active=False))
        for name, metrics in summary.items():
            session.add(ModelRun(
                created_at=created, batch=batch, name=name, title=model_titles()[name],
                freq_minutes=settings.freq_minutes, horizon_steps=horizon, metrics=metrics,
                artifact=str(artifact_path) if name == best else None, active=name == best,
            ))
    _artifact_cache.clear()
    return {"batch": batch, "best": best, "metrics": summary, "artifact": str(artifact_path),
            "zones": list(zone_names.values())}


def active_run(session: Session) -> Optional[ModelRun]:
    return session.scalar(select(ModelRun).where(ModelRun.active.is_(True)).order_by(ModelRun.id.desc()))


def load_artifact(path: str) -> Optional[dict]:
    file = Path(path)
    if not file.exists():
        return None
    stamp = file.stat().st_mtime
    cached = _artifact_cache.get(path)
    if cached is None or cached[0] != stamp:
        _artifact_cache[path] = (stamp, joblib.load(file))
    return _artifact_cache[path][1]


def refresh_forecasts(settings: Optional[Settings] = None, now: Optional[datetime] = None) -> int:
    """Пересчитать прогноз для всех зон активной моделью. Возвращает число записанных точек."""
    settings = settings or get_settings()
    now = now or utcnow()
    freq, horizon, spd = settings.freq_minutes, settings.horizon_steps, settings.steps_per_day
    end = floor_to(now, freq)  # начало текущего шага: он ещё не закончился и тоже прогнозируется
    written = 0
    with session_scope() as session:
        run = active_run(session)
        artifact = load_artifact(run.artifact) if run is not None and run.artifact else None
        if artifact is None:
            return 0
        horizon = min(horizon, artifact["horizon_steps"])
        for zone_id, pack in artifact["zones"].items():
            if session.get(Zone, zone_id) is None:
                continue
            data = zone_series(session, zone_id, settings, end=end, lookback_days=LOOKBACK_DAYS)
            if len(data) < 2 * spd:
                continue
            yhat = np.maximum(pack["model"].predict(data, len(data) - 1, horizon), 0.0)
            lo, hi = pack["intervals"].apply(yhat)
            # будущие точки прошлых расчётов больше не нужны, кроме контрольных горизонтов
            session.execute(delete(Forecast).where(
                Forecast.zone_id == zone_id, Forecast.ts >= end, Forecast.horizon.notin_(KEEP_HORIZONS)
            ))
            session.execute(delete(Forecast).where(
                Forecast.zone_id == zone_id, Forecast.created_at == now
            ))
            for step in range(horizon):
                session.add(Forecast(
                    zone_id=zone_id, run_id=run.id, created_at=now, ts=end + timedelta(minutes=freq * step),
                    horizon=step + 1, yhat=float(yhat[step]), lo=float(lo[step]), hi=float(hi[step]),
                ))
            written += horizon
        session.execute(delete(Forecast).where(Forecast.created_at < now - timedelta(days=90)))
    return written


def latest_forecast(session: Session, zone_id: int) -> List[Forecast]:
    created = session.scalar(select(func.max(Forecast.created_at)).where(Forecast.zone_id == zone_id))
    if created is None:
        return []
    return list(session.scalars(
        select(Forecast).where(Forecast.zone_id == zone_id, Forecast.created_at == created).order_by(Forecast.ts)
    ).all())
