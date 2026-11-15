import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from occupancy.api.app import create_app
from occupancy.forecast import service


@pytest.fixture()
def client(env):
    with TestClient(create_app(env, background=False)) as test_client:
        yield test_client


@pytest.fixture()
def admin(client):
    response = client.post("/admin/login", data={"password": "secret"}, follow_redirects=False)
    assert response.status_code == 303
    return client


@pytest.mark.parametrize("path", ["/", "/history", "/models", "/admin"])
def test_pages_open(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "Загруженность помещений" in response.text


def test_overview_without_data(client):
    data = client.get("/api/overview").json()
    assert [zone["name"] for zone in data["zones"]] == ["Первая зона", "Вторая зона", "Третья зона"]
    assert all(zone["count"] is None and zone["status"] == "unknown" for zone in data["zones"])
    assert data["total"] is None and data["capacity"] == 26 and data["model"] is None


def test_overview_timeline_and_week_with_data(client, zones, seeded):
    service.train(only=["weekly_profile"], include_neural=False)
    service.refresh_forecasts()
    overview = client.get("/api/overview").json()
    assert overview["model"]["title"] == "Типовой профиль недели"
    assert all(zone["count"] is not None for zone in overview["zones"])

    timeline = client.get(f"/api/zones/{zones[0].id}/timeline?back_hours=6").json()
    assert timeline["has_forecast"] and timeline["now_index"] == 24
    assert len(timeline["labels"]) == 24 + 96
    assert timeline["observed"][0] is not None and timeline["observed"][-1] is None
    assert timeline["forecast"][0] is None and timeline["forecast"][-1] is not None

    week = client.get(f"/api/zones/{zones[0].id}/week").json()
    assert len(week["cells"]) == 7 and len(week["cells"][0]) == 24
    assert client.get("/api/zones/999/timeline").status_code == 404


def test_history_and_export(client, seeded):
    history = client.get("/api/history").json()
    assert len(history["series"]) == 3 and history["step"] == 60
    assert len(history["labels"]) == len(history["series"][0]["values"]) == 7 * 24
    assert client.get("/api/history?start=2026-11-10&end=2026-11-01").status_code == 422

    csv = client.get("/api/export?format=csv&step=60")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    frame = pd.read_csv(io.BytesIO(csv.content))
    assert list(frame.columns) == ["Время", "Зона", "Людей", "Вместимость"] and len(frame) > 100

    xlsx = client.get("/api/export?format=xlsx&step=60")
    assert xlsx.status_code == 200
    assert len(pd.read_excel(io.BytesIO(xlsx.content))) == len(frame)
    assert client.get("/api/export?start=2020-01-01&end=2020-01-02").status_code == 404


def test_admin_api_requires_login(client):
    assert client.get("/api/admin/status").status_code == 401
    assert client.post("/api/admin/zones", json={"name": "Новая", "capacity": 5}).status_code == 401
    wrong = client.post("/admin/login", data={"password": "nope"})
    assert "Неверный пароль" in wrong.text
    assert client.get("/api/admin/status").status_code == 401


def test_zone_editing(admin):
    created = admin.post("/api/admin/zones", json={"name": "Переговорная", "capacity": 6,
                                                   "x1": 0.8, "y1": 0.9, "x2": 0.2, "y2": 0.1}).json()
    assert (created["x1"], created["y1"], created["x2"], created["y2"]) == (0.2, 0.1, 0.8, 0.9)
    assert admin.post("/api/admin/zones", json={"name": "Переговорная", "capacity": 6}).status_code == 409
    assert admin.post("/api/admin/zones", json={"name": "", "capacity": 0}).status_code == 422

    updated = admin.put(f"/api/admin/zones/{created['id']}", json={**created, "name": "Лаунж", "capacity": 12})
    assert updated.json()["name"] == "Лаунж" and updated.json()["capacity"] == 12

    assert admin.delete(f"/api/admin/zones/{created['id']}").json() == {"ok": True}
    assert "Лаунж" not in [zone["name"] for zone in admin.get("/api/zones").json()]


def test_source_settings(admin):
    status = admin.get("/api/admin/status").json()
    assert status["source"]["mode"] == "off" and status["source"]["state"] == "stopped"
    assert admin.put("/api/admin/source", json={"mode": "wrong"}).status_code == 422
    assert admin.put("/api/admin/source", json={"mode": "off", "url": "0"}).json() == {"mode": "off", "url": "0"}


def test_models_endpoint(client, seeded):
    empty = client.get("/api/models").json()
    assert empty["runs"] == [] and empty["data"]["rows"] > 0
    service.train(only=["seasonal_naive", "weekly_profile"], include_neural=False)
    runs = client.get("/api/models").json()["runs"]
    assert {run["name"] for run in runs} == {"seasonal_naive", "weekly_profile"}
    assert sum(run["active"] for run in runs) == 1
