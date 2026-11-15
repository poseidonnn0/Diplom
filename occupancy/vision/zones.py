"""Геометрия зон и подсчёт людей по зонам. Модуль не зависит от OpenCV и нейросети."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Detection:
    """Рамка найденного человека в пикселях кадра."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ZoneRect:
    """Зона в долях кадра: (0, 0) левый верхний угол, (1, 1) правый нижний."""

    id: int
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    capacity: int = 10
    color: str = "#0E4439"

    @classmethod
    def from_row(cls, zone) -> "ZoneRect":
        return cls(zone.id, zone.name, zone.x1, zone.y1, zone.x2, zone.y2, zone.capacity, zone.color)

    def contains(self, rx: float, ry: float) -> bool:
        return self.x1 <= rx <= self.x2 and self.y1 <= ry <= self.y2

    def pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (round(self.x1 * width), round(self.y1 * height), round(self.x2 * width), round(self.y2 * height))


def anchor_point(det: Detection, mode: str = "center") -> Tuple[float, float]:
    """Точка, по которой человек относится к зоне.

    center: центр рамки (как в бакалаврской версии);
    bottom: середина нижней стороны, то есть место, где человек стоит или сидит.
    """
    x = (det.x1 + det.x2) / 2
    y = det.y2 if mode == "bottom" else (det.y1 + det.y2) / 2
    return x, y


def count_by_zone(
    detections: Iterable[Detection],
    zones: Sequence[ZoneRect],
    width: int,
    height: int,
    anchor: str = "center",
) -> Dict[int, int]:
    """Сколько людей в каждой зоне. Человек относится к первой зоне, в которую попала его точка."""
    counts = {zone.id: 0 for zone in zones}
    if width <= 0 or height <= 0:
        return counts
    for det in detections:
        px, py = anchor_point(det, anchor)
        rx, ry = px / width, py / height
        for zone in zones:
            if zone.contains(rx, ry):
                counts[zone.id] += 1
                break
    return counts


def clamp_rect(x1: float, y1: float, x2: float, y2: float) -> List[float]:
    """Привести прямоугольник к виду x1 < x2, y1 < y2 в пределах 0..1."""
    xs = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    ys = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    return [xs[0], ys[0], xs[1], ys[1]]
