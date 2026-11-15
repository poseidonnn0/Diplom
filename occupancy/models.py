"""Таблицы базы данных."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Zone(Base):
    """Зона помещения: прямоугольник в долях кадра (0..1), не зависит от разрешения камеры."""

    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    x1: Mapped[float] = mapped_column(Float, default=0.0)
    y1: Mapped[float] = mapped_column(Float, default=0.0)
    x2: Mapped[float] = mapped_column(Float, default=1.0)
    y2: Mapped[float] = mapped_column(Float, default=1.0)
    capacity: Mapped[int] = mapped_column(Integer, default=10)
    color: Mapped[str] = mapped_column(String(9), default="#0E4439")
    position: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Observation(Base):
    """Число людей в зоне за одну минуту (медиана кадров этой минуты)."""

    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("zone_id", "ts", name="uq_observation_zone_ts"),
        Index("ix_observation_zone_ts", "zone_id", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(DateTime)  # UTC, начало минуты
    count: Mapped[int] = mapped_column(Integer)
    samples: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[str] = mapped_column(String(16), default="video")  # video, demo, import


class ModelRun(Base):
    """Результат обучения одной модели: метрики бэктеста и путь к сохранённой модели."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    batch: Mapped[str] = mapped_column(String(32), index=True)  # общий идентификатор запуска обучения
    name: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(80))
    freq_minutes: Mapped[int] = mapped_column(Integer)
    horizon_steps: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False)


class Forecast(Base):
    """Одна точка прогноза: зона, момент времени, значение и границы интервала."""

    __tablename__ = "forecasts"
    __table_args__ = (Index("ix_forecast_zone_created", "zone_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id", ondelete="CASCADE"))
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)  # UTC, момент расчёта
    ts: Mapped[datetime] = mapped_column(DateTime)  # UTC, на какое время прогноз
    horizon: Mapped[int] = mapped_column(Integer)  # на сколько шагов вперёд
    yhat: Mapped[float] = mapped_column(Float)
    lo: Mapped[float] = mapped_column(Float)
    hi: Mapped[float] = mapped_column(Float)


class Setting(Base):
    """Настройки, которые меняются из интерфейса (источник видео и т. п.)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
