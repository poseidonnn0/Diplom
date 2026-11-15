"""Подключение к базе данных и первичное наполнение."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base, Zone

_engine: Optional[Engine] = None
_factory: Optional[sessionmaker] = None

# Зоны по умолчанию повторяют бакалаврскую версию: кадр делится на три вертикальные полосы.
DEFAULT_ZONES = [
    {"name": "Первая зона", "x1": 0.0, "x2": 1 / 3, "capacity": 7, "color": "#3DBE8B"},
    {"name": "Вторая зона", "x1": 1 / 3, "x2": 2 / 3, "capacity": 9, "color": "#F08A5D"},
    {"name": "Третья зона", "x1": 2 / 3, "x2": 1.0, "capacity": 10, "color": "#6FA8F5"},
]


def get_engine() -> Engine:
    global _engine, _factory
    if _engine is None:
        url = get_settings().database_url
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" not in url:
                Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - настройка драйвера
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        _factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    """Закрыть подключение, чтобы следующее обращение создало его заново (для тестов)."""
    global _engine, _factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _factory is not None
    session = _factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(seed_zones: bool = True) -> None:
    """Создать таблицы и, если зон ещё нет, добавить три зоны по умолчанию."""
    Base.metadata.create_all(get_engine())
    if not seed_zones:
        return
    with session_scope() as session:
        if session.scalar(select(Zone.id).limit(1)) is None:
            for position, spec in enumerate(DEFAULT_ZONES):
                session.add(Zone(y1=0.0, y2=1.0, position=position, **spec))
