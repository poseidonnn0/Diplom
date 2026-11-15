import numpy as np
import pandas as pd

from occupancy.forecast.dataset import fill_gaps, make_data, prepare, to_regular

TZ = "Europe/Kaliningrad"  # UTC+2 круглый год


def minute_series(start, values):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="min"), dtype=float)


def test_to_regular_averages_minutes_into_local_steps():
    counts = minute_series("2026-11-15 10:00", [2] * 15 + [4] * 15)
    regular = to_regular(counts, 15, TZ)
    assert list(regular) == [2.0, 4.0]
    assert str(regular.index[0]) == "2026-11-15 12:00:00+02:00"


def test_to_regular_extends_grid_to_requested_end():
    counts = minute_series("2026-11-15 10:00", [3] * 15)
    regular = to_regular(counts, 15, TZ, end=pd.Timestamp("2026-11-15 11:00"))
    assert len(regular) == 4  # 12:00, 12:15, 12:30, 12:45 местного
    assert regular.iloc[0] == 3.0 and regular.iloc[1:].isna().all()


def test_fill_gaps_interpolates_short_and_uses_profile_for_long():
    index = pd.date_range("2026-11-02 00:00", periods=96 * 14, freq="15min", tz=TZ)
    series = pd.Series(np.tile(np.arange(96, dtype=float), 14), index=index)
    series.iloc[100:102] = np.nan  # короткий пропуск
    series.iloc[96 * 7 : 96 * 8] = np.nan  # целые сутки
    filled = fill_gaps(series, spd=96)
    assert not filled.isna().any()
    assert filled.iloc[100] == 4.0 and filled.iloc[101] == 5.0
    assert np.allclose(filled.iloc[96 * 7 : 96 * 8], np.arange(96))


def test_calendar_and_target_calendar():
    index = pd.date_range("2026-11-15 23:30", periods=4, freq="15min", tz=TZ)  # воскресенье -> понедельник
    data = make_data(pd.Series([1.0, 2.0, 3.0, 4.0], index=index), 15)
    assert list(data.slot) == [94, 95, 0, 1]
    assert list(data.dow) == [6, 6, 0, 0]
    slot, dow = data.target_calendar(np.array([0]), np.array([1, 2, 98]))
    assert slot.tolist() == [[95, 0, 0]] and dow.tolist() == [[6, 0, 1]]


def test_prepare_drops_leading_gap_and_returns_complete_series():
    counts = minute_series("2026-11-15 10:07", [1] * 120)
    data = prepare(counts, 15, TZ)
    assert len(data) == 9 and not np.isnan(data.y).any()
    assert prepare(pd.Series(dtype=float), 15, TZ).y.size == 0
