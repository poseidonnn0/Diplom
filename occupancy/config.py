"""Настройки приложения. Читаются из переменных окружения OCC_* и файла .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCC_", env_file=".env", extra="ignore")

    # Хранилище
    database_url: str = "sqlite:///" + (DATA_DIR / "occupancy.db").as_posix()
    artifacts_dir: Path = DATA_DIR / "models"
    timezone: str = "Europe/Kaliningrad"

    # Сбор подсчётов: off, demo или video
    source: str = "off"
    video_url: str = "data/samples/test_video.mp4"
    video_loop: bool = True
    weights: str = "yolov8n.pt"
    confidence: float = 0.5
    sample_seconds: float = 3.0
    anchor: str = "center"  # center или bottom: какой точкой рамки человек относится к зоне

    # Прогноз
    freq_minutes: int = 15
    horizon_steps: int = 96  # 96 шагов по 15 минут = сутки
    forecast_refresh_minutes: int = 15
    interval_coverage: float = 0.8
    backtest_days: int = 7

    # Администрирование
    admin_password: Optional[str] = None
    secret_key: Optional[str] = None

    @property
    def steps_per_day(self) -> int:
        return 24 * 60 // self.freq_minutes


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Сбросить кеш настроек (нужно в тестах после смены переменных окружения)."""
    get_settings.cache_clear()
