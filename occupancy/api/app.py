"""Веб-приложение: страницы интерфейса и JSON API."""
from __future__ import annotations

import asyncio
import hmac
import io
import logging
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware

from .. import analytics
from ..config import Settings, get_settings
from ..db import init_db, session_scope
from ..forecast import service
from ..models import ModelRun, Zone
from ..store import data_bounds
from ..timeutil import floor_to, to_local, utcnow
from ..vision.collector import CollectorManager
from ..vision.zones import clamp_rect

log = logging.getLogger("occupancy.api")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_EXPORT_DAYS = 92


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    capacity: int = Field(ge=1, le=10000)
    color: str = Field(default="#0E4439", pattern=r"^#[0-9A-Fa-f]{6}$")
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0


class SourceIn(BaseModel):
    mode: str
    url: Optional[str] = None


class TrainIn(BaseModel):
    include_neural: bool = True


def zone_dict(zone: Zone) -> dict:
    return {
        "id": zone.id, "name": zone.name, "capacity": zone.capacity, "color": zone.color,
        "x1": zone.x1, "y1": zone.y1, "x2": zone.x2, "y2": zone.y2, "position": zone.position,
    }


def create_app(settings: Optional[Settings] = None, background: bool = True) -> FastAPI:
    """Собрать приложение. background=False отключает сборщик и планировщик (нужно в тестах)."""
    settings = settings or get_settings()
    admin_password = settings.admin_password or secrets.token_urlsafe(9)
    manager = CollectorManager(settings)
    training = {"state": "idle", "message": None, "started": None, "finished": None}
    training_lock = threading.Lock()

    async def forecast_loop() -> None:
        """Раз в минуту проверять, не начался ли новый шаг, и пересчитывать прогноз."""
        last_step = None
        while True:
            try:
                step = floor_to(utcnow(), settings.forecast_refresh_minutes)
                if step != last_step and training["state"] != "running":
                    written = await asyncio.to_thread(service.refresh_forecasts, settings)
                    last_step = step
                    if written:
                        log.info("Прогноз обновлён: %s точек", written)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Не удалось обновить прогноз")
            await asyncio.sleep(60)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db()
        task = None
        if background:
            if not settings.admin_password:
                print(f"\n  Пароль администратора на этот запуск: {admin_password}\n"
                      "  Чтобы задать постоянный, укажите OCC_ADMIN_PASSWORD в файле .env\n", flush=True)
            manager.start()
            task = asyncio.create_task(forecast_loop())
        yield
        if task is not None:
            task.cancel()
        manager.stop()

    app = FastAPI(title="Прогноз загруженности помещений", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key or secrets.token_urlsafe(32),
                       session_cookie="occ_session", same_site="lax", max_age=12 * 3600)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")
    app.state.manager = manager
    app.state.training = training
    app.state.admin_password = admin_password

    def is_admin(request: Request) -> bool:
        return bool(request.session.get("admin"))

    def require_admin(request: Request) -> None:
        if not is_admin(request):
            raise HTTPException(status_code=401, detail="Нужен вход администратора")

    def page(request: Request, name: str, **context) -> HTMLResponse:
        context.update({"admin": is_admin(request), "active": name})
        return templates.TemplateResponse(request, f"{name}.html", context)

    def get_zone(zone_id: int) -> Zone:
        with session_scope() as session:
            zone = session.get(Zone, zone_id)
            if zone is None:
                raise HTTPException(status_code=404, detail="Зона не найдена")
            return zone

    def parse_period(start: Optional[date], end: Optional[date], limit_days: int) -> tuple:
        today = to_local(utcnow(), settings.timezone).date()
        end = end or today
        start = start or end - timedelta(days=6)
        if start > end:
            raise HTTPException(status_code=422, detail="Начало периода позже конца")
        if (end - start).days + 1 > limit_days:
            raise HTTPException(status_code=422, detail=f"Период не длиннее {limit_days} дней")
        return start, end

    # ---------- страницы ----------
    @app.get("/", response_class=HTMLResponse)
    def dashboard_page(request: Request):
        return page(request, "dashboard")

    @app.get("/history", response_class=HTMLResponse)
    def history_page(request: Request):
        return page(request, "history")

    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request):
        return page(request, "models")

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page(request: Request):
        if not is_admin(request):
            return page(request, "login", error=None)
        return page(request, "admin")

    @app.post("/admin/login")
    async def admin_login(request: Request):
        form = await request.form()
        password = str(form.get("password", ""))
        if hmac.compare_digest(password.encode(), admin_password.encode()):
            request.session["admin"] = True
            return RedirectResponse("/admin", status_code=303)
        return page(request, "login", error="Неверный пароль")

    @app.post("/admin/logout")
    def admin_logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=303)

    # ---------- открытое API ----------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "time": utcnow().isoformat()}

    @app.get("/api/overview")
    def api_overview():
        with session_scope() as session:
            return analytics.overview(session, settings, manager.info(), utcnow())

    @app.get("/api/zones")
    def api_zones():
        with session_scope() as session:
            return [zone_dict(zone) for zone in analytics.active_zones(session)]

    @app.get("/api/zones/{zone_id}/timeline")
    def api_timeline(zone_id: int, back_hours: int = Query(12, ge=1, le=72)):
        zone = get_zone(zone_id)
        with session_scope() as session:
            return analytics.timeline(session, zone, settings, utcnow(), back_hours)

    @app.get("/api/zones/{zone_id}/week")
    def api_week(zone_id: int, weeks: int = Query(8, ge=1, le=52)):
        zone = get_zone(zone_id)
        with session_scope() as session:
            return analytics.weekly_heatmap(session, zone, settings, utcnow(), weeks)

    @app.get("/api/history")
    def api_history(start: Optional[date] = None, end: Optional[date] = None):
        start, end = parse_period(start, end, 366)
        with session_scope() as session:
            data = analytics.history(session, settings, start, end)
        data.update({"start": start.isoformat(), "end": end.isoformat()})
        return data

    @app.get("/api/export")
    def api_export(start: Optional[date] = None, end: Optional[date] = None,
                   step: int = Query(1, ge=1, le=1440), format: str = Query("xlsx", pattern="^(xlsx|csv)$")):
        start, end = parse_period(start, end, MAX_EXPORT_DAYS)
        with session_scope() as session:
            frame = analytics.export_frame(session, settings, start, end, step)
        if frame.empty:
            raise HTTPException(status_code=404, detail="За этот период данных нет")
        name = f"occupancy_{start.isoformat()}_{end.isoformat()}.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{name}"'}
        if format == "csv":
            body = frame.to_csv(index=False).encode("utf-8-sig")  # BOM, чтобы Excel правильно показал кириллицу
            return Response(body, media_type="text/csv; charset=utf-8", headers=headers)
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False, sheet_name="Посещаемость")
        return Response(buffer.getvalue(), headers=headers,
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.get("/api/models")
    def api_models():
        with session_scope() as session:
            batch = session.scalar(select(ModelRun.batch).order_by(ModelRun.id.desc()).limit(1))
            runs = session.scalars(select(ModelRun).where(ModelRun.batch == batch).order_by(ModelRun.id)).all() if batch else []
            first, last, total = data_bounds(session)
            return {
                "trained": to_local(runs[0].created_at, settings.timezone).strftime("%d.%m.%Y %H:%M") if runs else None,
                "freq_minutes": settings.freq_minutes,
                "horizon_steps": settings.horizon_steps,
                "coverage_target": settings.interval_coverage,
                "runs": [{"name": run.name, "title": run.title, "active": run.active, "metrics": run.metrics} for run in runs],
                "data": {
                    "rows": total,
                    "start": to_local(first, settings.timezone).strftime("%d.%m.%Y") if first else None,
                    "end": to_local(last, settings.timezone).strftime("%d.%m.%Y") if last else None,
                },
                "training": dict(training),
            }

    # ---------- администрирование ----------
    @app.get("/api/admin/status", dependencies=[Depends(require_admin)])
    def api_status():
        info = manager.info()
        if info.get("last_update") is not None:
            info["last_update"] = to_local(info["last_update"], settings.timezone).strftime("%H:%M:%S")
        return {"source": info, "training": dict(training)}

    @app.put("/api/admin/source", dependencies=[Depends(require_admin)])
    def api_source(body: SourceIn):
        try:
            return manager.set_config(body.mode, body.url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/admin/frame.jpg", dependencies=[Depends(require_admin)])
    def api_frame():
        collector = manager.collector
        jpeg = getattr(collector, "last_jpeg", None)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="Кадров пока нет")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/admin/zones", dependencies=[Depends(require_admin)])
    def api_zone_create(body: ZoneIn):
        with session_scope() as session:
            if session.scalar(select(Zone).where(Zone.name == body.name)) is not None:
                raise HTTPException(status_code=409, detail="Зона с таким названием уже есть")
            x1, y1, x2, y2 = clamp_rect(body.x1, body.y1, body.x2, body.y2)
            position = len(session.scalars(select(Zone.id)).all())
            zone = Zone(name=body.name, capacity=body.capacity, color=body.color,
                        x1=x1, y1=y1, x2=x2, y2=y2, position=position)
            session.add(zone)
            session.flush()
            return zone_dict(zone)

    @app.put("/api/admin/zones/{zone_id}", dependencies=[Depends(require_admin)])
    def api_zone_update(zone_id: int, body: ZoneIn):
        with session_scope() as session:
            zone = session.get(Zone, zone_id)
            if zone is None:
                raise HTTPException(status_code=404, detail="Зона не найдена")
            clash = session.scalar(select(Zone).where(Zone.name == body.name, Zone.id != zone_id))
            if clash is not None:
                raise HTTPException(status_code=409, detail="Зона с таким названием уже есть")
            zone.name, zone.capacity, zone.color = body.name, body.capacity, body.color
            zone.x1, zone.y1, zone.x2, zone.y2 = clamp_rect(body.x1, body.y1, body.x2, body.y2)
            return zone_dict(zone)

    @app.delete("/api/admin/zones/{zone_id}", dependencies=[Depends(require_admin)])
    def api_zone_delete(zone_id: int):
        """Зона скрывается, а не удаляется: накопленная по ней история остаётся в базе."""
        with session_scope() as session:
            zone = session.get(Zone, zone_id)
            if zone is None:
                raise HTTPException(status_code=404, detail="Зона не найдена")
            zone.active = False
            zone.name = f"{zone.name} (архив {zone.id})"[:80]
        return {"ok": True}

    @app.post("/api/admin/train", dependencies=[Depends(require_admin)])
    def api_train(body: TrainIn):
        with training_lock:
            if training["state"] == "running":
                raise HTTPException(status_code=409, detail="Обучение уже идёт")
            training.update({"state": "running", "message": "Сравниваю модели на истории",
                             "started": to_local(utcnow(), settings.timezone).strftime("%H:%M:%S"), "finished": None})

        def job() -> None:
            try:
                result = service.train(settings, include_neural=body.include_neural)
                service.refresh_forecasts(settings)
                training.update({"state": "done", "message": f"Лучшая модель: {service.model_titles()[result['best']]}"})
            except Exception as exc:
                log.exception("Обучение не удалось")
                training.update({"state": "error", "message": str(exc)})
            finally:
                training["finished"] = to_local(utcnow(), settings.timezone).strftime("%H:%M:%S")

        threading.Thread(target=job, daemon=True, name="training").start()
        return dict(training)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    return app
