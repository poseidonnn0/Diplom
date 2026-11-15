"""Сбор подсчётов: кадр -> люди -> числа по зонам -> одна запись в минуту на зону."""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select

from ..config import Settings
from ..db import session_scope
from ..demo import live_count
from ..models import Observation, Setting, Zone
from ..timeutil import floor_minute, utcnow
from .zones import ZoneRect, count_by_zone

log = logging.getLogger("occupancy.collector")

Row = Tuple[datetime, int, int, int]  # минута, зона, число людей, сколько кадров усреднено


class MinuteAggregator:
    """Копит измерения внутри минуты и отдаёт медиану, когда минута закончилась.

    Медиана гасит одиночные ошибки детектора: пропущенного на одном кадре человека
    или случайное ложное срабатывание.
    """

    def __init__(self) -> None:
        self._minute: Optional[datetime] = None
        self._samples: Dict[int, List[int]] = {}

    def add(self, ts: datetime, counts: Dict[int, int]) -> List[Row]:
        minute = floor_minute(ts)
        finished: List[Row] = []
        if self._minute is not None and minute != self._minute:
            finished = self.flush()
        self._minute = minute
        for zone_id, count in counts.items():
            self._samples.setdefault(zone_id, []).append(int(count))
        return finished

    def flush(self) -> List[Row]:
        if self._minute is None:
            return []
        rows = [
            (self._minute, zone_id, int(round(median(values))), len(values))
            for zone_id, values in self._samples.items()
            if values
        ]
        self._minute = None
        self._samples = {}
        return rows


def save_observations(rows: List[Row], source: str) -> None:
    """Записать минутные значения. Повторная запись той же минуты обновляет строку."""
    if not rows:
        return
    with session_scope() as session:
        for ts, zone_id, count, samples in rows:
            existing = session.scalar(
                select(Observation).where(Observation.zone_id == zone_id, Observation.ts == ts)
            )
            if existing is None:
                session.add(Observation(zone_id=zone_id, ts=ts, count=count, samples=samples, source=source))
            else:
                existing.count, existing.samples, existing.source = count, samples, source


def load_zones() -> List[ZoneRect]:
    with session_scope() as session:
        rows = session.scalars(select(Zone).where(Zone.active.is_(True)).order_by(Zone.position, Zone.id)).all()
        return [ZoneRect.from_row(row) for row in rows]


class BaseCollector(threading.Thread):
    """Общий цикл: раз в sample_seconds получить подсчёт и передать его в агрегатор."""

    source_name = "video"
    zones_ttl = 5.0  # как часто перечитывать зоны из базы, секунды

    def __init__(self, settings: Settings):
        super().__init__(daemon=True, name="collector")
        self.settings = settings
        self.state = "starting"
        self.error: Optional[str] = None
        self.last_counts: Dict[int, int] = {}
        self.last_update: Optional[datetime] = None
        self.last_jpeg: Optional[bytes] = None
        self.frame_id = 0
        self._halt = threading.Event()
        self._aggregator = MinuteAggregator()
        self._zones: List[ZoneRect] = []
        self._zones_loaded: Optional[datetime] = None

    # --- переопределяется в наследниках ---
    def prepare(self) -> None:
        """Тяжёлая инициализация (загрузка нейросети), выполняется уже в рабочем потоке."""

    def sample(self, now: datetime) -> Optional[Dict[int, int]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # --- общая часть ---
    def zones(self, now: datetime) -> List[ZoneRect]:
        if self._zones_loaded is None or (now - self._zones_loaded).total_seconds() > self.zones_ttl:
            self._zones = load_zones()
            self._zones_loaded = now
        return self._zones

    def stop(self) -> None:
        self._halt.set()

    def info(self) -> dict:
        return {
            "mode": self.source_name,
            "state": self.state,
            "error": self.error,
            "last_update": self.last_update,
            "counts": dict(self.last_counts),
        }

    def run(self) -> None:
        try:
            self.prepare()
        except Exception as exc:
            log.exception("Сборщик не запустился")
            self.state, self.error = "error", str(exc)
            return
        while not self._halt.is_set():
            now = utcnow()
            try:
                counts = self.sample(now)
            except Exception as exc:  # одна неудача не должна останавливать сбор
                log.exception("Ошибка при обработке кадра")
                self.state, self.error = "error", str(exc)
                counts = None
            if counts is not None:
                self.state, self.error = "running", None
                self.last_counts, self.last_update = counts, now
                save_observations(self._aggregator.add(now, counts), self.source_name)
            self._halt.wait(self.settings.sample_seconds)
        save_observations(self._aggregator.flush(), self.source_name)
        self.close()
        self.state = "stopped"


class DemoCollector(BaseCollector):
    """Имитация камеры: числа берутся из типового профиля дня с шумом."""

    source_name = "demo"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._rng = np.random.default_rng()

    def sample(self, now: datetime) -> Dict[int, int]:
        zones = self.zones(now)
        return {
            zone.id: live_count(zone, index, now, self.settings.timezone, self._rng)
            for index, zone in enumerate(zones)
        }


class VideoCollector(BaseCollector):
    """Реальный источник: файл, USB-камера или сетевой поток плюс детектор YOLO."""

    source_name = "video"

    def __init__(self, settings: Settings, url: str):
        super().__init__(settings)
        self.url = url
        self.frame_size = (0, 0)
        self._grabber = None
        self._detector = None
        self._seen_frame = -1

    def prepare(self) -> None:  # pragma: no cover - требует OpenCV и весов нейросети
        from .detector import PersonDetector
        from .sources import FrameGrabber

        self._detector = PersonDetector(self.settings.weights, self.settings.confidence)
        self._grabber = FrameGrabber(self.url, loop=self.settings.video_loop)
        self._grabber.start()

    def sample(self, now: datetime) -> Optional[Dict[int, int]]:  # pragma: no cover
        frame, frame_id = self._grabber.latest()
        if frame is None or frame_id == self._seen_frame:
            if self._grabber.error:
                self.state, self.error = "error", self._grabber.error
            return None
        self._seen_frame = frame_id
        height, width = frame.shape[:2]
        self.frame_size = (width, height)
        zones = self.zones(now)
        detections = self._detector.detect(frame)
        counts = count_by_zone(detections, zones, width, height, self.settings.anchor)
        self.last_jpeg = self._annotate(frame, detections)
        self.frame_id += 1
        return counts

    def _annotate(self, frame, detections) -> Optional[bytes]:  # pragma: no cover
        import cv2

        from .zones import anchor_point

        # Рамки людей рисуются на кадре, а границы и названия зон накладывает браузер:
        # так они сразу отражают правки в редакторе и не зависят от шрифтов OpenCV.
        canvas = frame.copy()
        for det in detections:
            p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
            cv2.rectangle(canvas, p1, p2, (36, 178, 255), 2)
            ax, ay = anchor_point(det, self.settings.anchor)
            cv2.circle(canvas, (int(ax), int(ay)), 5, (36, 178, 255), -1)
        ok, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ok else None

    def close(self) -> None:  # pragma: no cover
        if self._grabber is not None:
            self._grabber.stop()

    def info(self) -> dict:
        data = super().info()
        data.update({"url": self.url, "frame_size": list(self.frame_size), "has_video": self.last_jpeg is not None})
        return data


class CollectorManager:
    """Запускает и перезапускает сборщик. Режим берётся из базы, а если там пусто, из настроек."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector: Optional[BaseCollector] = None
        self._lock = threading.Lock()

    def config(self) -> dict:
        with session_scope() as session:
            stored = session.get(Setting, "source")
            if stored is not None and stored.value.get("mode") in ("off", "demo", "video"):
                return {"mode": stored.value["mode"], "url": stored.value.get("url") or self.settings.video_url}
        return {"mode": self.settings.source, "url": self.settings.video_url}

    def set_config(self, mode: str, url: Optional[str]) -> dict:
        if mode not in ("off", "demo", "video"):
            raise ValueError("Режим должен быть off, demo или video")
        with session_scope() as session:
            stored = session.get(Setting, "source")
            value = {"mode": mode, "url": (url or "").strip() or self.settings.video_url}
            if stored is None:
                session.add(Setting(key="source", value=value))
            else:
                stored.value = value
        self.restart()
        return self.config()

    def start(self) -> None:
        with self._lock:
            if self.collector is not None and self.collector.is_alive():
                return
            config = self.config()
            if config["mode"] == "demo":
                self.collector = DemoCollector(self.settings)
            elif config["mode"] == "video":
                self.collector = VideoCollector(self.settings, config["url"])
            else:
                self.collector = None
                return
            self.collector.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            collector, self.collector = self.collector, None
        if collector is not None:
            collector.stop()
            collector.join(timeout)

    def restart(self) -> None:
        self.stop()
        self.start()

    def info(self) -> dict:
        config = self.config()
        collector = self.collector
        if collector is None:
            return {"mode": config["mode"], "url": config["url"], "state": "stopped", "error": None,
                    "last_update": None, "counts": {}}
        data = collector.info()
        data.setdefault("url", config["url"])
        return data
