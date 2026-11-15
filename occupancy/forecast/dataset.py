"""Подготовка ряда посещаемости: равномерная сетка, заполнение пропусков, календарные признаки."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SeriesData:
    """Ряд на равномерной сетке и календарь к нему.

    y: среднее число людей в зоне за шаг;
    slot: номер шага внутри суток (0 .. spd-1);
    dow: день недели (0 = понедельник);
    spd: число шагов в сутках.
    """

    y: np.ndarray
    slot: np.ndarray
    dow: np.ndarray
    spd: int
    index: Optional[pd.DatetimeIndex] = None

    def __len__(self) -> int:
        return len(self.y)

    def head(self, n: int) -> "SeriesData":
        index = self.index[:n] if self.index is not None else None
        return SeriesData(self.y[:n], self.slot[:n], self.dow[:n], self.spd, index)

    def target_calendar(self, origins: np.ndarray, horizons: np.ndarray):
        """Шаг суток и день недели для моментов origin + h (матрицы origins x horizons)."""
        ahead = self.slot[origins][:, None] + horizons[None, :]
        slot = ahead % self.spd
        dow = (self.dow[origins][:, None] + ahead // self.spd) % 7
        return slot, dow


def to_regular(
    counts: pd.Series,
    freq_minutes: int,
    tz: str,
    end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Минутные подсчёты (UTC) -> средние по шагам freq_minutes в местном времени.

    Шаги без данных получают NaN. Если задан end (UTC), сетка доводится до последнего
    полного шага перед end, даже если данных там нет.
    """
    if counts.empty:
        return pd.Series(dtype=float)
    local = counts.copy()
    local.index = pd.DatetimeIndex(local.index).tz_localize("UTC").tz_convert(tz)
    rule = f"{freq_minutes}min"
    regular = local.resample(rule).mean()
    if end is not None:
        last = pd.Timestamp(end).tz_localize("UTC").tz_convert(tz).floor(rule) - pd.Timedelta(minutes=freq_minutes)
        if last > regular.index[-1]:
            regular = regular.reindex(pd.date_range(regular.index[0], last, freq=rule))
        else:
            regular = regular[:last]
    return regular


def fill_gaps(series: pd.Series, spd: int, short_limit: int = 4) -> pd.Series:
    """Заполнить пропуски: короткие линейно, длинные типовым значением для этого дня и времени."""
    if series.empty:
        return series
    missing = series.isna()
    run = (missing != missing.shift()).cumsum()
    short = missing & (missing.groupby(run).transform("sum") <= short_limit)
    # линейно заполняются только пропуски целиком не длиннее short_limit шагов
    filled = series.where(~short, series.interpolate(limit_area="inside"))
    if filled.isna().any():
        slot = (filled.index.hour * 60 + filled.index.minute) * spd // (24 * 60)
        keys = pd.DataFrame({"dow": filled.index.weekday, "slot": slot, "y": filled.to_numpy()}, index=filled.index)
        by_dow_slot = keys.groupby(["dow", "slot"])["y"].transform("mean")
        by_slot = keys.groupby("slot")["y"].transform("mean")
        filled = filled.fillna(by_dow_slot).fillna(by_slot).fillna(0.0)
    return filled


def make_data(series: pd.Series, freq_minutes: int) -> SeriesData:
    """Собрать SeriesData из ряда на равномерной сетке без пропусков."""
    spd = 24 * 60 // freq_minutes
    index = pd.DatetimeIndex(series.index)
    slot = ((index.hour * 60 + index.minute) // freq_minutes).to_numpy().astype(int)
    return SeriesData(
        y=series.to_numpy(dtype=float),
        slot=slot,
        dow=index.weekday.to_numpy().astype(int),
        spd=spd,
        index=index,
    )


def prepare(counts: pd.Series, freq_minutes: int, tz: str, end: Optional[pd.Timestamp] = None) -> SeriesData:
    """Полный путь от минутных подсчётов до данных для моделей."""
    spd = 24 * 60 // freq_minutes
    regular = to_regular(counts, freq_minutes, tz, end)
    if regular.empty:
        return SeriesData(np.array([]), np.array([], dtype=int), np.array([], dtype=int), spd)
    first = regular.first_valid_index()
    regular = regular[first:]
    return make_data(fill_gaps(regular, spd), freq_minutes)
