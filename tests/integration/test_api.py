from datetime import date

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.main import app, provide_curve
from yieldcurve.bootstrap import build_curve


@pytest.fixture
def curve(settlements_fixture):
    return build_curve(settlements_fixture)


@pytest.fixture
def client(curve):
    app.dependency_overrides[provide_curve] = lambda: curve
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_curve_meta(client, curve):
    r = client.get("/curve")
    assert r.status_code == 200
    body = r.json()
    du, _ = curve.node_rates()
    assert body["d0"] == curve.d0.isoformat()
    assert body["interpolation"] == "flat_forward"
    assert body["n_nodes"] == du.size
    assert body["du_min"] == int(du[0]) == 1
    assert body["du_max"] == int(du[-1])


def test_curve_nodes(client, curve):
    r = client.get("/curve/nodes")
    assert r.status_code == 200
    nodes = r.json()
    du, zero = curve.node_rates()
    assert len(nodes) == du.size
    assert nodes[0]["du"] == int(du[0])
    assert nodes[0]["zero"] == pytest.approx(float(zero[0]), abs=1e-10)
    assert nodes[0]["df"] == pytest.approx(curve.df(int(du[0])), abs=1e-10)


def test_df_by_du(client, curve):
    r = client.get("/curve/df", params={"du": 252})
    assert r.status_code == 200
    assert r.json() == {"du": 252, "value": pytest.approx(curve.df(252), abs=1e-12)}


def test_df_by_date(client, curve):
    d = date(2027, 1, 4)
    r = client.get("/curve/df", params={"date": d.isoformat()})
    assert r.status_code == 200
    body = r.json()
    assert body["du"] == curve.du(d)
    assert body["value"] == pytest.approx(curve.df(d), abs=1e-12)


def test_df_requires_a_term(client):
    r = client.get("/curve/df")
    assert r.status_code == 422


def test_zero_by_du(client, curve):
    r = client.get("/curve/zero", params={"du": 252})
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(curve.zero(252), abs=1e-12)


def test_zero_undefined_at_du_zero(client):
    r = client.get("/curve/zero", params={"du": 0})
    assert r.status_code == 422


def test_forward(client, curve):
    r = client.get("/curve/forward", params={"du1": 252, "du2": 504})
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "du1": 252,
        "du2": 504,
        "forward": pytest.approx(curve.forward(252, 504), abs=1e-12),
    }


def test_forward_requires_ordered_terms(client):
    r = client.get("/curve/forward", params={"du1": 504, "du2": 252})
    assert r.status_code == 422


def test_forward_missing_side(client):
    r = client.get("/curve/forward", params={"du1": 252})
    assert r.status_code == 422


def test_risk_dv01(client, curve):
    from yieldcurve.risk import dv01, key_rate_dv01

    payload = {"cashflows": [{"du": 252, "amount": 1_000_000}, {"du": 504, "amount": -400_000}]}
    r = client.post("/risk/dv01", json=payload)
    assert r.status_code == 200
    body = r.json()
    cfs = [(252, 1_000_000.0), (504, -400_000.0)]
    assert body["dv01"] == pytest.approx(dv01(curve, cfs), rel=1e-9)
    kr = key_rate_dv01(curve, cfs)
    assert body["total_key_rate"] == pytest.approx(kr.total, rel=1e-9)
    assert body["total_key_rate"] == pytest.approx(body["dv01"], rel=1e-3)
    du, _ = curve.node_rates()
    assert len(body["key_rate"]) == du.size


def test_risk_rejects_empty_portfolio(client):
    r = client.post("/risk/dv01", json={"cashflows": []})
    assert r.status_code == 422


def test_risk_rejects_nonpositive_du(client):
    r = client.post("/risk/dv01", json={"cashflows": [{"du": 0, "amount": 1.0}]})
    assert r.status_code == 422
