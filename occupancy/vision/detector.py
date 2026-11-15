"""Детектор людей на YOLO. Тяжёлые библиотеки подключаются только при создании детектора."""
from __future__ import annotations

from typing import List, Optional

from .zones import Detection

PERSON_CLASS = 0  # номер класса «человек» в наборе COCO


class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.5, device: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для видеоаналитики установите зависимости: pip install -r requirements-vision.txt"
            ) from exc
        self.model = YOLO(weights)  # веса скачиваются автоматически при первом запуске
        self.confidence = confidence
        self.device = device

    def detect(self, frame) -> List[Detection]:
        """Найти людей на кадре (массив BGR, как отдаёт OpenCV)."""
        result = self.model.predict(
            frame, classes=[PERSON_CLASS], conf=self.confidence, device=self.device, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            Detection(float(x1), float(y1), float(x2), float(y2), float(score))
            for (x1, y1, x2, y2), score in zip(boxes, scores)
        ]
