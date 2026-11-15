"""Работа со временем: в базе хранится UTC без часового пояса, пользователю показывается местное время."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def utcnow() -> datetime:
    """Текущее время в UTC без tzinfo (в таком виде оно лежит в базе)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_local(ts: datetime, tz: str) -> datetime:
    """UTC из базы -> местное время с часовым поясом."""
    return ts.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))


def to_utc(ts: datetime, tz: str) -> datetime:
    """Местное время (наивное считается местным) -> UTC без tzinfo."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo(tz))
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def floor_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def floor_to(ts: datetime, minutes: int) -> datetime:
    """Округлить вниз до шага в minutes минут (от начала суток)."""
    ts = floor_minute(ts)
    return ts - timedelta(minutes=(ts.hour * 60 + ts.minute) % minutes)
