"""Источники кадров: файл, USB-камера, сетевой поток (RTSP, HLS)."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional


def describe_source(url: str) -> str:
    if url.isdigit():
        return "usb"
    if "://" in url:
        return "stream"
    return "file"


def open_capture(url: str):
    """Открыть источник через OpenCV. Возвращает cv2.VideoCapture."""
    import cv2

    kind = describe_source(url)
    if kind == "usb":
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        return cv2.VideoCapture(int(url), backend)
    if kind == "stream":
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(url)


class FrameGrabber(threading.Thread):
    """Читает источник в отдельном потоке и хранит только последний кадр.

    Для камер и сетевых потоков это избавляет от накопления задержки в буфере.
    Файл проигрывается с его собственной частотой кадров и при loop=True запускается заново.
    При обрыве связи источник переоткрывается с нарастающей паузой.
    """

    def __init__(self, url: str, loop: bool = True):
        super().__init__(daemon=True, name="frame-grabber")
        self.url = url
        self.loop = loop
        self.kind = describe_source(url)
        self.error: Optional[str] = None
        self.size = (0, 0)
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._halt = threading.Event()

    def latest(self):
        """Последний кадр и его порядковый номер (None, если кадров ещё не было)."""
        with self._lock:
            return self._frame, self._frame_id

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:  # pragma: no cover - требует OpenCV и реального источника
        delay = 1.0
        while not self._halt.is_set():
            cap = open_capture(self.url)
            if not cap.isOpened():
                self.error = f"Не удалось открыть источник: {self.url}"
                cap.release()
                self._halt.wait(delay)
                delay = min(delay * 2, 30.0)
                continue
            self.error = None
            delay = 1.0
            import cv2

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_time = 1.0 / fps if self.kind == "file" and 1.0 < fps < 121.0 else 0.0
            while not self._halt.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self.size = (frame.shape[1], frame.shape[0])
                if frame_time:
                    self._halt.wait(max(0.0, frame_time - (time.monotonic() - started)))
            cap.release()
            if self.kind == "file" and not self.loop:
                break
            if self.kind != "file":
                self.error = "Связь с источником потеряна, переподключение"
                self._halt.wait(delay)
