from datetime import datetime

from sqlalchemy import select

from occupancy.db import session_scope
from occupancy.models import Observation
from occupancy.vision.collector import DemoCollector, MinuteAggregator, save_observations


def test_aggregator_returns_median_when_minute_changes():
    aggregator = MinuteAggregator()
    minute = datetime(2026, 11, 15, 12, 30)
    assert aggregator.add(minute.replace(second=1), {1: 4, 2: 0}) == []
    assert aggregator.add(minute.replace(second=20), {1: 9, 2: 0}) == []  # одиночный выброс
    assert aggregator.add(minute.replace(second=40), {1: 5, 2: 1}) == []
    rows = aggregator.add(minute.replace(minute=31), {1: 6, 2: 1})
    assert rows == [(minute, 1, 5, 3), (minute, 2, 0, 3)]
    assert aggregator.flush() == [(minute.replace(minute=31), 1, 6, 1), (minute.replace(minute=31), 2, 1, 1)]
    assert aggregator.flush() == []


def test_saving_same_minute_twice_updates_row(env, zones):
    minute = datetime(2026, 11, 15, 12, 30)
    save_observations([(minute, zones[0].id, 3, 10)], "video")
    save_observations([(minute, zones[0].id, 5, 20)], "video")
    with session_scope() as session:
        rows = session.scalars(select(Observation)).all()
    assert [(row.count, row.samples) for row in rows] == [(5, 20)]


def test_demo_collector_gives_plausible_counts(env, zones):
    collector = DemoCollector(env)
    midday = datetime(2026, 11, 16, 10, 30)  # понедельник, 12:30 по Калининграду
    night = datetime(2026, 11, 16, 1, 0)  # 03:00, закрыто
    day_counts = collector.sample(midday)
    assert set(day_counts) == {zone.id for zone in zones}
    assert all(0 <= day_counts[zone.id] <= zone.capacity + 2 for zone in zones)
    assert sum(day_counts.values()) > 0
    assert collector.sample(night) == {zone.id: 0 for zone in zones}
