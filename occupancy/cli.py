"""Команды запуска: python -m occupancy <команда>."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


def cmd_init_db(_args) -> None:
    from .db import init_db

    init_db()
    print("База данных готова.")


def cmd_serve(args) -> None:
    import uvicorn

    if args.source:
        os.environ["OCC_SOURCE"] = args.source
    from .api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def cmd_demo_data(args) -> None:
    from sqlalchemy import select

    from .config import get_settings
    from .db import init_db, session_scope
    from .demo import generate_history
    from .models import Zone
    from .store import clear_source, insert_observations
    from .timeutil import utcnow

    settings = get_settings()
    init_db()
    if args.clear:
        print(f"Удалено демо-записей: {clear_source('demo')}")
        return
    with session_scope() as session:
        zones = session.scalars(select(Zone).where(Zone.active.is_(True)).order_by(Zone.position, Zone.id)).all()
    frame = generate_history(zones, args.days, utcnow(), settings.timezone, seed=args.seed)
    written = insert_observations(frame, source="demo")
    print(f"Записано {written} демо-записей за {args.days} сут. по {len(zones)} зонам.")


def cmd_import_gym(args) -> None:
    from .datasets import import_gym
    from .db import init_db

    init_db()
    info = import_gym(Path(args.path), zone_name=args.zone, capacity=args.capacity)
    print(f"Зона «{info['zone']}»: {info['rows']} записей с {info['start']} по {info['end']} (UTC), "
          f"вместимость {info['capacity']}.")


def cmd_train(args) -> None:
    from .config import get_settings
    from .db import init_db
    from .forecast import service

    init_db()
    result = service.train(get_settings(), include_neural=not args.no_neural, only=args.only, report=print)
    print()
    print(f"{'Модель':28s} {'MAE':>7s} {'RMSE':>7s} {'WAPE':>7s} {'Покрытие':>9s}")
    for name, metrics in sorted(result["metrics"].items(), key=lambda item: item[1]["mae"]):
        mark = " *" if name == result["best"] else ""
        coverage = f"{metrics['coverage'] * 100:6.1f} %" if "coverage" in metrics else "      —"
        print(f"{service.model_titles()[name]:28s} {metrics['mae']:7.3f} {metrics['rmse']:7.3f} "
              f"{metrics['wape'] * 100:6.1f}% {coverage}{mark}")
    print(f"\n* используется для прогноза. Модель сохранена: {result['artifact']}")
    print(f"Прогноз обновлён, точек: {service.refresh_forecasts(get_settings())}")


def cmd_forecast(_args) -> None:
    from .config import get_settings
    from .db import init_db
    from .forecast import service

    init_db()
    written = service.refresh_forecasts(get_settings())
    print(f"Записано точек прогноза: {written}" if written else "Прогноз не построен: нет обученной модели или истории.")


def cmd_collect(args) -> None:
    from .config import get_settings
    from .db import init_db
    from .vision.collector import CollectorManager

    if args.source:
        os.environ["OCC_SOURCE"] = args.source
    init_db()
    manager = CollectorManager(get_settings())
    manager.start()
    if manager.collector is None:
        sys.exit("Источник выключен. Задайте OCC_SOURCE=demo или OCC_SOURCE=video.")
    print("Сбор запущен. Остановить: Ctrl+C.")
    try:
        while True:
            time.sleep(10)
            info = manager.info()
            print(f"{info['state']:9s} {info.get('counts')} {info.get('error') or ''}")
    except KeyboardInterrupt:
        manager.stop()
        print("Сбор остановлен, последняя минута записана.")


def cmd_count_video(args) -> None:
    """Прогнать видеофайл через детектор и показать подсчёты по зонам (проверка без сервера)."""
    import cv2

    from .config import get_settings
    from .db import init_db
    from .vision.collector import load_zones
    from .vision.detector import PersonDetector
    from .vision.zones import count_by_zone

    settings = get_settings()
    init_db()
    zones = load_zones()
    detector = PersonDetector(args.weights or settings.weights, settings.confidence)
    capture = cv2.VideoCapture(args.path)
    if not capture.isOpened():
        sys.exit(f"Не удалось открыть видео: {args.path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    every = max(1, int(round(fps * args.every)))
    print("Время, с  " + "  ".join(f"{zone.name:>12s}" for zone in zones) + "   Всего")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            height, width = frame.shape[:2]
            detections = detector.detect(frame)
            counts = count_by_zone(detections, zones, width, height, settings.anchor)
            print(f"{index / fps:8.1f}  " + "  ".join(f"{counts[zone.id]:12d}" for zone in zones)
                  + f"   {len(detections):5d}")
        index += 1
    capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m occupancy",
                                     description="Интеллектуальная система прогнозирования загруженности помещений")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="создать таблицы и зоны по умолчанию").set_defaults(func=cmd_init_db)

    serve = sub.add_parser("serve", help="запустить веб-сервис")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--source", choices=["off", "demo", "video"], help="источник подсчётов на этот запуск")
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("demo-data", help="заполнить базу имитацией истории")
    demo.add_argument("--days", type=int, default=45)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--clear", action="store_true", help="удалить демо-записи вместо добавления")
    demo.set_defaults(func=cmd_demo_data)

    gym = sub.add_parser("import-gym", help="загрузить открытый набор Crowdedness at the Campus Gym (CSV)")
    gym.add_argument("path")
    gym.add_argument("--zone", default="Campus Gym")
    gym.add_argument("--capacity", type=int)
    gym.set_defaults(func=cmd_import_gym)

    train = sub.add_parser("train", help="сравнить модели, обучить лучшую и обновить прогноз")
    train.add_argument("--no-neural", action="store_true", help="не обучать нейросеть (быстрее на процессоре)")
    train.add_argument("--only", nargs="+", help="имена моделей, например: weekly_profile gradient_boosting")
    train.set_defaults(func=cmd_train)

    sub.add_parser("forecast", help="пересчитать прогноз активной моделью").set_defaults(func=cmd_forecast)

    collect = sub.add_parser("collect", help="собирать подсчёты без веб-сервиса")
    collect.add_argument("--source", choices=["demo", "video"])
    collect.set_defaults(func=cmd_collect)

    video = sub.add_parser("count-video", help="проверить детектор на видеофайле")
    video.add_argument("path")
    video.add_argument("--every", type=float, default=1.0, help="шаг между кадрами, секунды")
    video.add_argument("--weights")
    video.set_defaults(func=cmd_count_video)
    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)
