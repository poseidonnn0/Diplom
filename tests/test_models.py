import numpy as np
import pytest

from occupancy.forecast.dataset import SeriesData
from occupancy.forecast.evaluate import HorizonIntervals, backtest
from occupancy.forecast.models import (
    GradientBoosting, HorizonBlend, RecurrentNetwork, SeasonalNaive, WeeklyProfile, candidates, torch_available,
    weekly_profile,
)

SPD = 96


def periodic(days: int) -> SeriesData:
    """Идеально повторяющаяся неделя: значение зависит только от дня недели и шага суток."""
    steps = np.arange(days * SPD)
    slot, dow = steps % SPD, (steps // SPD) % 7
    y = 5 + 4 * np.sin(2 * np.pi * slot / SPD) + dow
    return SeriesData(y=y.astype(float), slot=slot, dow=dow, spd=SPD)


def mae(model, data, horizon=SPD):
    origins = np.arange(len(data) - 3 * SPD, len(data) - horizon, 8)
    predictions = model.predict_many(data, origins, horizon)
    actual = data.y[origins[:, None] + np.arange(1, horizon + 1)[None, :]]
    return float(np.mean(np.abs(predictions - actual)))


def test_weekly_profile_matrix():
    profile = weekly_profile(periodic(14))
    assert profile.shape == (7, SPD)
    assert profile[3, 0] == pytest.approx(8.0)  # 5 + sin(0) + день недели 3


def test_naive_and_profile_are_exact_on_periodic_series():
    data = periodic(21)
    assert mae(SeasonalNaive().fit(data), data) == pytest.approx(0.0, abs=1e-9)
    assert mae(WeeklyProfile().fit(data), data) == pytest.approx(0.0, abs=1e-9)


def test_naive_falls_back_to_yesterday_without_a_full_week():
    data = periodic(3)
    prediction = SeasonalNaive().predict(data, len(data) - 1, 4)
    assert np.allclose(prediction, data.y[len(data) - SPD : len(data) - SPD + 4])


@pytest.mark.parametrize("factory", [SeasonalNaive, WeeklyProfile, GradientBoosting, HorizonBlend])
def test_models_do_not_look_into_the_future(factory, demo_data):
    """Прогноз из точки origin не должен меняться, если испортить значения после неё."""
    model = factory().fit(demo_data.head(len(demo_data) - 2 * SPD))
    origin = len(demo_data) - 2 * SPD
    before = model.predict(demo_data, origin, SPD)
    spoiled = SeriesData(demo_data.y.copy(), demo_data.slot, demo_data.dow, demo_data.spd)
    spoiled.y[origin + 1 :] = 999.0
    assert np.allclose(before, model.predict(spoiled, origin, SPD))
    assert before.shape == (SPD,) and np.all(np.isfinite(before)) and np.all(before >= 0)


def test_boosting_beats_naive_on_demo_series(demo_data):
    train = demo_data.head(len(demo_data) - 3 * SPD)
    assert mae(GradientBoosting().fit(train), demo_data) < mae(SeasonalNaive().fit(train), demo_data)


def test_blend_weights_are_valid(demo_data):
    model = HorizonBlend().fit(demo_data)
    assert model.weights.shape == (SPD,)
    assert np.all((model.weights >= 0) & (model.weights <= 1))


@pytest.mark.skipif(not torch_available(), reason="torch не установлен")
def test_recurrent_network_trains_and_survives_pickling(demo_data):
    import pickle

    model = RecurrentNetwork(horizon=SPD, epochs=1, hidden=16).fit(demo_data.head(10 * SPD))
    origin = 10 * SPD - 1
    prediction = model.predict(demo_data, origin, SPD)
    assert prediction.shape == (SPD,) and np.all(prediction >= 0)
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(prediction, restored.predict(demo_data, origin, SPD), atol=1e-5)


def test_candidates_registry():
    names = set(candidates(include_neural=False))
    assert names == {"seasonal_naive", "weekly_profile", "gradient_boosting", "horizon_blend"}


def test_intervals_reach_requested_coverage():
    rng = np.random.default_rng(0)
    errors = np.abs(rng.normal(0, 1 + np.linspace(0, 1, 24), size=(400, 24)))  # дальше горизонт, шире ошибка
    intervals = HorizonIntervals(0.8).fit(errors[:200])
    assert intervals.q[-1] > intervals.q[0]
    lo, hi = intervals.apply(np.full((200, 24), 5.0))
    inside = (5.0 - errors[200:] >= lo) & (5.0 + errors[200:] <= hi)
    assert 0.7 < inside.mean() < 0.9
    assert np.all(intervals.apply(np.zeros(24))[0] == 0)  # нижняя граница не уходит ниже нуля


def test_backtest_reports_metrics(demo_data):
    result = backtest(WeeklyProfile(), demo_data, horizon=SPD, test_steps=5 * SPD)
    assert result.predictions.shape == result.actual.shape == (len(result.origins), SPD)
    assert set(result.metrics["by_horizon"]) == {"до 1 часа", "1–6 часов", "6–24 часа"}
    assert result.metrics["mae"] <= result.metrics["rmse"]
    assert 0 <= result.metrics["coverage"] <= 1
    with pytest.raises(ValueError):
        backtest(WeeklyProfile(), demo_data.head(3 * SPD), horizon=SPD, test_steps=2 * SPD)
