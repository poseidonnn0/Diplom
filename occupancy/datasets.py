"""Импорт открытых наборов данных о посещаемости.

Сейчас поддержан набор «Crowdedness at the Campus Gym» (Kaggle): число людей в зале
примерно каждые 10 минут в течение года. Нужны колонки number_people и date.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select

from .db import session_scope
from .models import Zone
from .store import insert_observations


def read_gym_csv(path: Path) -> pd.DataFrame:
    """Прочитать CSV набора и вернуть колонки ts (UTC, без пояса) и count."""
    raw = pd.read_csv(path)
    missing = {"number_people", "date"} - set(raw.columns)
    if missing:
        raise ValueError(f"В файле нет колонок: {', '.join(sorted(missing))}")
    # В наборе время записано с местным смещением (например, 2015-08-14 17:00:11-07:00).
    ts = pd.to_datetime(raw["date"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = pd.DataFrame({"ts": ts, "count": pd.to_numeric(raw["number_people"], errors="coerce")})
    return frame.dropna().sort_values("ts").reset_index(drop=True)


def import_gym(path: Path, zone_name: str = "Campus Gym", capacity: Optional[int] = None) -> dict:
    """Загрузить набор как отдельную зону. Вместимость по умолчанию берётся как 95-й процентиль."""
    frame = read_gym_csv(Path(path))
    if frame.empty:
        raise ValueError("В файле нет ни одной корректной строки")
    if capacity is None:
        capacity = int(max(1, round(frame["count"].quantile(0.95))))
    with session_scope() as session:
        zone = session.scalar(select(Zone).where(Zone.name == zone_name))
        if zone is None:
            position = len(session.scalars(select(Zone.id)).all())
            zone = Zone(name=zone_name, capacity=capacity, color="#3B7DD8", position=position)
            session.add(zone)
            session.flush()
        else:
            zone.capacity = capacity
        zone_id = zone.id
    frame["zone_id"] = zone_id
    written = insert_observations(frame, source="import")
    return {
        "zone_id": zone_id,
        "zone": zone_name,
        "rows": written,
        "capacity": capacity,
        "start": frame["ts"].min(),
        "end": frame["ts"].max(),
    }
