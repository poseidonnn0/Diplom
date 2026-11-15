from datetime import datetime
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import func, select

from occupancy.datasets import import_gym
from occupancy.db import session_scope
from occupancy.demo import generate_history
from occupancy.models import Observation, Zone
from occupancy.store import clear_source, insert_observations


def test_demo_history_respects_opening_hours_and_capacity():
    zone = SimpleNamespace(id=1, capacity=10)
    frame = generate_history([zone], 7, datetime(2026, 11, 15, 22, 0), "Europe/Kaliningrad", seed=5)
    assert len(frame) == 7 * 24 * 60
    local_hour = (frame["ts"] + pd.Timedelta(hours=2)).dt.hour
    assert frame.loc[(local_hour < 8) | (local_hour >= 22), "count"].sum() == 0
    assert frame["count"].between(0, 12).all() and frame["count"].max() >= 5


def test_insert_is_idempotent_and_clear_removes_only_one_source(env, zones):
    frame = generate_history(zones[:1], 1, datetime(2026, 11, 15, 22, 0), env.timezone)
    assert insert_observations(frame, "demo") == 1440
    assert insert_observations(frame, "demo") == 1440  # повтор не создаёт дублей
    other = pd.DataFrame({"ts": [datetime(2026, 1, 1, 12, 0)], "zone_id": [zones[1].id], "count": [3]})
    insert_observations(other, "video")
    with session_scope() as session:
        assert session.scalar(select(func.count(Observation.id))) == 1441
    assert clear_source("demo") == 1440
    with session_scope() as session:
        assert session.scalar(select(func.count(Observation.id))) == 1


def test_import_gym_csv(env, tmp_path):
    path = tmp_path / "gym.csv"
    pd.DataFrame({
        "number_people": [37, 45, 40, 44],
        "date": ["2015-08-14 17:00:11-07:00", "2015-08-14 17:20:14-07:00",
                 "2015-08-14 17:30:15-07:00", "2015-08-14 17:40:16-07:00"],
        "temperature": [71.76, 71.76, 71.76, 71.76],
    }).to_csv(path, index=False)
    info = import_gym(path)
    assert info["rows"] == 4 and info["start"] == pd.Timestamp("2015-08-15 00:00:11")
    import_gym(path)  # повторный импорт обновляет ту же зону
    with session_scope() as session:
        zone = session.scalar(select(Zone).where(Zone.name == "Campus Gym"))
        rows = session.scalars(select(Observation).where(Observation.zone_id == zone.id)).all()
    assert zone.capacity == info["capacity"] and len(rows) == 4
    assert {row.source for row in rows} == {"import"}
