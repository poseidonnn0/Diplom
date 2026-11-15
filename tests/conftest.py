import os
from datetime import datetime

import pytest
from sqlalchemy import select


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Чистая база и настройки во временной папке."""
    monkeypatch.setenv("OCC_DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("OCC_ARTIFACTS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("OCC_ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("OCC_SOURCE", "off")
    monkeypatch.setenv("OCC_TIMEZONE", "Europe/Kaliningrad")
    from occupancy import config, db

    config.reset_settings()
    db.reset_engine()
    db.init_db()
    yield config.get_settings()
    db.reset_engine()
    config.reset_settings()


@pytest.fixture()
def zones(env):
    from occupancy.db import session_scope
    from occupancy.models import Zone

    with session_scope() as session:
        return session.scalars(select(Zone).order_by(Zone.position)).all()


@pytest.fixture()
def seeded(env, zones):
    """База с 16 сутками имитации, история заканчивается «сейчас»."""
    from occupancy.demo import generate_history
    from occupancy.store import insert_observations
    from occupancy.timeutil import utcnow

    now = utcnow()
    insert_observations(generate_history(zones, 16, now, env.timezone, seed=1), source="demo")
    return now


@pytest.fixture(scope="session")
def demo_data():
    """Ряд для проверки моделей: 21 сутки имитации одной зоны, шаг 15 минут."""
    from types import SimpleNamespace

    from occupancy.demo import generate_history
    from occupancy.forecast.dataset import prepare

    zone = SimpleNamespace(id=1, capacity=12)
    frame = generate_history([zone], 21, datetime(2026, 11, 15, 22, 0), "Europe/Kaliningrad", seed=3)
    counts = frame.set_index("ts")["count"].astype(float)
    return prepare(counts, 15, "Europe/Kaliningrad")
