"""Имитация посещаемости: история для первого запуска и «живые» подсчёты без камеры.

Форма дня: помещение открыто с 8 до 22, дневной пик около полудня и второй пик ближе к вечеру.
Будни загружены сильнее выходных, у каждой зоны свой масштаб и сдвиг по времени.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OPEN_HOUR = 8
CLOSE_HOUR = 22
WEEKDAY_FACTOR = np.array([1.0, 1.0, 0.97, 1.0, 0.85, 0.55, 0.35])  # пн..вс
ZONE_SCALE = (0.92, 0.78, 0.62)
ZONE_SHIFT = (0.0, 0.4, -0.5)  # сдвиг пиков в часах


def load_fraction(hours: np.ndarray, weekday: np.ndarray, zone_index: int) -> np.ndarray:
    """Ожидаемая доля занятых мест для времени суток (в часах, дробное) и дня недели (0 = пн)."""
    hours = np.asarray(hours, dtype=float)
    weekday = np.asarray(weekday, dtype=int)
    shift = ZONE_SHIFT[zone_index % len(ZONE_SHIFT)]
    scale = ZONE_SCALE[zone_index % len(ZONE_SCALE)]
    h = hours - shift
    shape = 0.08 + 0.80 * np.exp(-(((h - 12.4) / 1.55) ** 2)) + 0.62 * np.exp(-(((h - 17.1) / 1.5) ** 2))
    # плавное заполнение после открытия и опустение перед закрытием
    ramp = np.clip((hours - OPEN_HOUR) / 0.75, 0, 1) * np.clip((CLOSE_HOUR - hours) / 0.75, 0, 1)
    return np.clip(shape * ramp * scale * WEEKDAY_FACTOR[weekday], 0, 1)


def generate_history(
    zones: Sequence, days: int, end_utc: datetime, tz: str, seed: int = 42
) -> pd.DataFrame:
    """Поминутная история за days суток до end_utc. Колонки: ts (UTC), zone_id, count."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_utc).floor("min")
    index_utc = pd.date_range(end - pd.Timedelta(days=days), end, freq="min", inclusive="left")
    local = index_utc.tz_localize("UTC").tz_convert(ZoneInfo(tz))
    hours = (local.hour + local.minute / 60).to_numpy()
    weekday = local.weekday.to_numpy()
    local_days = local.tz_localize(None).normalize()
    day_number = (local_days - local_days[0]).days.to_numpy()

    frames: List[pd.DataFrame] = []
    for zone_index, zone in enumerate(zones):
        expected = load_fraction(hours, weekday, zone_index) * zone.capacity
        # каждый день немного загруженнее или свободнее обычного
        day_factor = rng.lognormal(mean=0.0, sigma=0.16, size=int(day_number.max()) + 1)[day_number]
        # медленный шум внутри дня: соседние минуты похожи друг на друга
        noise = np.empty(len(index_utc))
        rho, value = 0.985, 0.0
        steps = rng.normal(0.0, 0.25 * np.sqrt(1 - rho**2), size=len(index_utc))
        for i, step in enumerate(steps):
            value = rho * value + step
            noise[i] = value
        level = expected * day_factor * np.exp(noise) + rng.normal(0.0, 0.35, size=len(index_utc))
        counts = np.clip(np.rint(level), 0, zone.capacity + 2).astype(int)
        counts[expected <= 0.01] = 0
        frames.append(pd.DataFrame({"ts": index_utc, "zone_id": zone.id, "count": counts}))
    return pd.concat(frames, ignore_index=True)


def live_count(zone, zone_index: int, now_utc: datetime, tz: str, rng: np.random.Generator) -> int:
    """Одно «измерение» для режима имитации: ожидаемая загрузка в данный момент плюс шум."""
    local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz))
    hour = local.hour + local.minute / 60 + local.second / 3600
    expected = float(load_fraction(np.array([hour]), np.array([local.weekday()]), zone_index)[0]) * zone.capacity
    if expected <= 0.01:
        return 0
    return int(np.clip(round(expected * rng.lognormal(0.0, 0.12) + rng.normal(0.0, 0.5)), 0, zone.capacity + 2))
