"""Данные для интерфейса: текущая загрузка, лента «факт и прогноз», история, типовая неделя."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .forecast.service import active_run, latest_forecast
from .models import Forecast, Observation, Zone
from .store import load_counts
from .timeutil import floor_to, to_local, to_utc

FREE_BELOW = 0.5  # до половины мест занято: «свободно»
BUSY_FROM = 0.85  # от 85 %: «занято»
FRESH_MINUTES = 15  # старше этого последнее измерение уже не считается текущим


def status_for(load: Optional[float]) -> str:
    if load is None:
        return "unknown"
    if load < FREE_BELOW:
        return "free"
    if load < BUSY_FROM:
        return "moderate"
    return "busy"


def active_zones(session: Session) -> List[Zone]:
    return list(session.scalars(select(Zone).where(Zone.active.is_(True)).order_by(Zone.position, Zone.id)).all())


def current_counts(session: Session, zones: Sequence[Zone], live: dict, now: datetime) -> Dict[int, tuple]:
    """Текущее число людей по зонам: из работающего сборщика или из последних записей в базе."""
    result: Dict[int, tuple] = {}
    if live.get("state") == "running" and live.get("last_update") is not None:
        for zone_id, count in live.get("counts", {}).items():
            result[int(zone_id)] = (int(count), live["last_update"])
    since = now - timedelta(minutes=FRESH_MINUTES)
    for zone in zones:
        if zone.id in result:
            continue
        row = session.execute(
            select(Observation.count, Observation.ts)
            .where(Observation.zone_id == zone.id, Observation.ts >= since, Observation.ts <= now)
            .order_by(Observation.ts.desc())
            .limit(1)
        ).first()
        if row is not None:
            result[zone.id] = (int(row[0]), row[1])
    return result


def insight(rows: Sequence[Forecast], capacity: int, now: datetime, settings: Settings) -> Optional[dict]:
    """Короткие выводы из прогноза: когда станет свободнее, когда ждать пика."""
    step = timedelta(minutes=settings.freq_minutes)
    future = [row for row in rows if row.ts + step > now]
    if not future or capacity <= 0:
        return None
    loads = np.array([row.yhat / capacity for row in future])
    horizon_12h = min(len(future), 12 * 60 // settings.freq_minutes)
    peak = int(np.argmax(loads[:horizon_12h]))
    result = {
        "peak_time": to_local(future[peak].ts, settings.timezone).strftime("%H:%M"),
        "peak_value": round(float(future[peak].yhat), 1),
        "free_time": None,
        "busy_time": None,
    }
    if loads[0] >= FREE_BELOW:
        free = np.nonzero(loads < FREE_BELOW)[0]
        if len(free):
            result["free_time"] = to_local(future[int(free[0])].ts, settings.timezone).strftime("%H:%M")
    if loads[0] < BUSY_FROM:
        busy = np.nonzero(loads[:horizon_12h] >= BUSY_FROM)[0]
        if len(busy):
            result["busy_time"] = to_local(future[int(busy[0])].ts, settings.timezone).strftime("%H:%M")
    return result


def overview(session: Session, settings: Settings, live: dict, now: datetime) -> dict:
    zones = active_zones(session)
    counts = current_counts(session, zones, live, now)
    cards = []
    for zone in zones:
        count, updated = counts.get(zone.id, (None, None))
        load = count / zone.capacity if count is not None and zone.capacity else None
        cards.append({
            "id": zone.id,
            "name": zone.name,
            "capacity": zone.capacity,
            "color": zone.color,
            "count": count,
            "load": round(load, 3) if load is not None else None,
            "status": status_for(load),
            "updated": to_local(updated, settings.timezone).strftime("%H:%M:%S") if updated else None,
            "insight": insight(latest_forecast(session, zone.id), zone.capacity, now, settings),
        })
    known = [card["count"] for card in cards if card["count"] is not None]
    run = active_run(session)
    return {
        "now": to_local(now, settings.timezone).strftime("%d.%m.%Y %H:%M"),
        "zones": cards,
        "total": sum(known) if known else None,
        "capacity": sum(zone.capacity for zone in zones),
        "source": {"mode": live.get("mode"), "state": live.get("state"), "error": live.get("error")},
        "model": {"title": run.title, "trained": to_local(run.created_at, settings.timezone).strftime("%d.%m.%Y %H:%M")}
        if run is not None else None,
    }


def timeline(session: Session, zone: Zone, settings: Settings, now: datetime, back_hours: int = 12) -> dict:
    """Лента по шагам прогноза: факт за последние часы и прогноз с интервалом вперёд."""
    freq = settings.freq_minutes
    step = timedelta(minutes=freq)
    end = floor_to(now, freq)
    start = end - timedelta(hours=back_hours)
    counts = load_counts(session, zone.id, start=start, end=end)
    observed = counts.resample(f"{freq}min").mean() if not counts.empty else pd.Series(dtype=float)
    forecast = {row.ts: row for row in latest_forecast(session, zone.id) if row.ts >= end}
    last = max(forecast) if forecast else end
    labels, days, fact, yhat, lo, hi = [], [], [], [], [], []
    moment = start
    while moment <= last:
        local = to_local(moment, settings.timezone)
        labels.append(local.strftime("%H:%M"))
        days.append(local.strftime("%d.%m"))
        value = observed.get(pd.Timestamp(moment)) if moment < end else None
        fact.append(round(float(value), 2) if value is not None and not pd.isna(value) else None)
        row = forecast.get(moment)
        yhat.append(round(row.yhat, 2) if row else None)
        lo.append(round(row.lo, 2) if row else None)
        hi.append(round(row.hi, 2) if row else None)
        moment += step
    return {
        "zone": {"id": zone.id, "name": zone.name, "capacity": zone.capacity, "color": zone.color},
        "labels": labels, "days": days, "observed": fact, "forecast": yhat, "lo": lo, "hi": hi,
        "now_index": int((end - start) / step),
        "freq_minutes": freq,
        "has_forecast": bool(forecast),
    }


def weekly_heatmap(session: Session, zone: Zone, settings: Settings, now: datetime, weeks: int = 8) -> dict:
    """Типовая неделя: средняя загрузка по дням недели и часам за последние недели."""
    counts = load_counts(session, zone.id, start=now - timedelta(weeks=weeks), end=now)
    cells: List[List[Optional[float]]] = [[None] * 24 for _ in range(7)]
    if not counts.empty:
        local = counts.copy()
        local.index = pd.DatetimeIndex(local.index).tz_localize("UTC").tz_convert(settings.timezone)
        grouped = local.groupby([local.index.weekday, local.index.hour]).mean()
        for (weekday, hour), value in grouped.items():
            cells[int(weekday)][int(hour)] = round(float(value) / max(zone.capacity, 1), 3)
    return {"zone": zone.name, "capacity": zone.capacity, "cells": cells, "weeks": weeks}


def day_bounds(start: date, end: date, tz: str) -> tuple:
    """Границы периода в UTC: с начала первого дня до конца последнего (по местному времени)."""
    return to_utc(datetime.combine(start, time.min), tz), to_utc(datetime.combine(end + timedelta(days=1), time.min), tz)


def pick_step(start: date, end: date) -> int:
    days = (end - start).days + 1
    if days <= 2:
        return 15
    if days <= 14:
        return 60
    return 24 * 60


def history(session: Session, settings: Settings, start: date, end: date, step: Optional[int] = None) -> dict:
    """История по всем зонам за период с шагом step минут и краткой статистикой."""
    step = step or pick_step(start, end)
    lo, hi = day_bounds(start, end, settings.timezone)
    grid = pd.date_range(
        pd.Timestamp(lo).tz_localize("UTC").tz_convert(settings.timezone),
        pd.Timestamp(hi).tz_localize("UTC").tz_convert(settings.timezone),
        freq=f"{step}min", inclusive="left",
    )
    fmt = "%d.%m" if step >= 24 * 60 else "%d.%m %H:%M"
    series, stats = [], []
    for zone in active_zones(session):
        counts = load_counts(session, zone.id, start=lo, end=hi)
        if counts.empty:
            values = [None] * len(grid)
            stats.append({"zone": zone.name, "mean": None, "peak": None, "peak_time": None, "busy_share": None})
        else:
            local = counts.copy()
            local.index = pd.DatetimeIndex(local.index).tz_localize("UTC").tz_convert(settings.timezone)
            resampled = local.resample(f"{step}min").mean().reindex(grid)
            values = [round(float(v), 2) if not pd.isna(v) else None for v in resampled]
            peak_at = local.idxmax()
            stats.append({
                "zone": zone.name,
                "mean": round(float(local.mean()), 2),
                "peak": int(local.max()),
                "peak_time": peak_at.strftime("%d.%m %H:%M"),
                "busy_share": round(float((local >= BUSY_FROM * zone.capacity).mean()), 3),
            })
        series.append({"id": zone.id, "name": zone.name, "color": zone.color, "capacity": zone.capacity, "values": values})
    return {"labels": [moment.strftime(fmt) for moment in grid], "series": series, "stats": stats, "step": step}


def export_frame(session: Session, settings: Settings, start: date, end: date, step: int = 1) -> pd.DataFrame:
    """Таблица для выгрузки: время, зона, число людей, вместимость."""
    lo, hi = day_bounds(start, end, settings.timezone)
    parts = []
    for zone in active_zones(session):
        counts = load_counts(session, zone.id, start=lo, end=hi)
        if counts.empty:
            continue
        local = counts.copy()
        local.index = pd.DatetimeIndex(local.index).tz_localize("UTC").tz_convert(settings.timezone)
        if step > 1:
            local = local.resample(f"{step}min").mean().dropna().round(2)
        parts.append(pd.DataFrame({
            "Время": local.index.tz_localize(None),
            "Зона": zone.name,
            "Людей": local.to_numpy(),
            "Вместимость": zone.capacity,
        }))
    if not parts:
        return pd.DataFrame(columns=["Время", "Зона", "Людей", "Вместимость"])
    return pd.concat(parts, ignore_index=True).sort_values(["Время", "Зона"]).reset_index(drop=True)
