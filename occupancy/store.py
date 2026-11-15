"""Чтение и массовая запись подсчётов."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from .db import session_scope
from .models import Observation


def insert_observations(frame: pd.DataFrame, source: str, replace: bool = True) -> int:
    """Массово записать подсчёты. frame: колонки ts (UTC), zone_id, count.

    При replace=True старые записи тех же зон за тот же промежуток сначала удаляются,
    поэтому повторный импорт не создаёт дублей.
    """
    if frame.empty:
        return 0
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"]).dt.floor("min")
    frame = frame.groupby(["zone_id", "ts"], as_index=False)["count"].median()
    with session_scope() as session:
        if replace:
            for zone_id, part in frame.groupby("zone_id"):
                session.execute(
                    delete(Observation).where(
                        Observation.zone_id == int(zone_id),
                        Observation.ts >= part["ts"].min().to_pydatetime(),
                        Observation.ts <= part["ts"].max().to_pydatetime(),
                    )
                )
        records = [
            {"zone_id": int(z), "ts": t.to_pydatetime(), "count": int(round(c)), "samples": 1, "source": source}
            for z, t, c in zip(frame["zone_id"], frame["ts"], frame["count"])
        ]
        for start in range(0, len(records), 5000):
            session.execute(insert(Observation), records[start : start + 5000])
    return len(records)


def clear_source(source: str) -> int:
    with session_scope() as session:
        result = session.execute(delete(Observation).where(Observation.source == source))
        return result.rowcount or 0


def load_counts(
    session: Session, zone_id: int, start: Optional[datetime] = None, end: Optional[datetime] = None
) -> pd.Series:
    """Минутные подсчёты зоны: индекс ts (UTC, без пояса), значения count."""
    query = select(Observation.ts, Observation.count).where(Observation.zone_id == zone_id)
    if start is not None:
        query = query.where(Observation.ts >= start)
    if end is not None:
        query = query.where(Observation.ts < end)
    rows = session.execute(query.order_by(Observation.ts)).all()
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="ts"), name="count")
    index = pd.DatetimeIndex([row[0] for row in rows], name="ts")
    return pd.Series([float(row[1]) for row in rows], index=index, name="count")


def data_bounds(session: Session, zone_id: Optional[int] = None):
    """Первая и последняя минута с данными и число записей."""
    query = select(func.min(Observation.ts), func.max(Observation.ts), func.count(Observation.id))
    if zone_id is not None:
        query = query.where(Observation.zone_id == zone_id)
    return session.execute(query).one()
