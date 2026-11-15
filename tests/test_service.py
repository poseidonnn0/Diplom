from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from occupancy.db import session_scope
from occupancy.forecast import service
from occupancy.models import Forecast, ModelRun


def test_train_requires_history(env):
    with pytest.raises(ValueError):
        service.train(env, include_neural=False)


def test_train_picks_best_model_and_builds_forecast(env, zones, seeded):
    result = service.train(env, include_neural=False, only=["seasonal_naive", "weekly_profile", "gradient_boosting"])
    assert result["best"] in result["metrics"]
    assert Path(result["artifact"]).exists()
    best_mae = result["metrics"][result["best"]]["mae"]
    assert all(best_mae <= metrics["mae"] for metrics in result["metrics"].values())
    assert result["metrics"]["seasonal_naive"]["skill"] == pytest.approx(0.0)

    with session_scope() as session:
        runs = session.scalars(select(ModelRun)).all()
        assert len(runs) == 3 and sum(run.active for run in runs) == 1

    horizon = env.horizon_steps
    assert service.refresh_forecasts(env, now=seeded) == horizon * len(zones)
    with session_scope() as session:
        rows = service.latest_forecast(session, zones[0].id)
    assert len(rows) == horizon
    assert [row.horizon for row in rows] == list(range(1, horizon + 1))
    assert rows[1].ts - rows[0].ts == timedelta(minutes=env.freq_minutes)
    assert all(row.lo <= row.yhat <= row.hi and row.lo >= 0 for row in rows)


def test_refresh_keeps_only_control_horizons_of_old_forecasts(env, zones, seeded):
    service.train(env, include_neural=False, only=["weekly_profile"])
    service.refresh_forecasts(env, now=seeded)
    later = seeded + timedelta(minutes=env.freq_minutes)
    service.refresh_forecasts(env, now=later)
    with session_scope() as session:
        old = session.scalars(
            select(Forecast.horizon).where(Forecast.zone_id == zones[0].id, Forecast.created_at == seeded)
        ).all()
        latest = session.scalar(select(func.count(Forecast.id)).where(
            Forecast.zone_id == zones[0].id, Forecast.created_at == later))
    # от прошлого расчёта остались контрольные горизонты и первая точка, которая уже в прошлом
    assert set(old) <= set(service.KEEP_HORIZONS) | {1}
    assert latest == env.horizon_steps


def test_new_training_replaces_active_model(env, zones, seeded):
    service.train(env, include_neural=False, only=["seasonal_naive"])
    service.train(env, include_neural=False, only=["weekly_profile"])
    with session_scope() as session:
        active = session.scalars(select(ModelRun).where(ModelRun.active.is_(True))).all()
    assert [run.name for run in active] == ["weekly_profile"]
