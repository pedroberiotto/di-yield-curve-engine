import csv
from datetime import date

import numpy as np
import pytest

from yieldcurve.bootstrap import Fixture, build_curve, load_settlements
from yieldcurve.instruments import DI1, rate_from_pu


def test_load_settlements_shape_and_anchor(settlements_fixture):
    fx = settlements_fixture
    assert fx.du.ndim == 1 and fx.du.size > 10
    assert fx.du.shape == fx.rate.shape
    assert np.all(np.diff(fx.du) > 0)
    assert fx.du[0] == 1
    assert np.all((fx.rate > 0) & (fx.rate < 1))


def test_load_settlements_spot_comes_from_pu(settlements_path):
    fx = load_settlements(settlements_path)
    with open(settlements_path) as fh:
        rows = list(csv.DictReader(fh))
    by_du = {int(fx.du[i]): float(fx.rate[i]) for i in range(fx.du.size)}
    for r in rows:
        du = int(r["business_days"])
        assert by_du[du] == pytest.approx(rate_from_pu(float(r["settlement_pu"]), du), abs=1e-12)


def test_load_settlements_explicit_overnight_rate(settlements_path):
    fx = load_settlements(settlements_path, overnight_rate=0.1390)
    assert fx.du[0] == 1
    assert fx.rate[0] == pytest.approx(0.1390, abs=1e-12)


def test_load_settlements_reparse_stable(settlements_path):
    a, b = load_settlements(settlements_path), load_settlements(settlements_path)
    assert a.d0 == b.d0
    assert np.array_equal(a.du, b.du)
    assert np.allclose(a.rate, b.rate)


def test_build_curve_passes_through_every_node(settlements_fixture):
    curve = build_curve(settlements_fixture)
    assert curve.df(0) == pytest.approx(1.0, abs=1e-15)
    for du, rate in zip(settlements_fixture.du, settlements_fixture.rate, strict=True):
        assert curve.zero(int(du)) == pytest.approx(rate, abs=1e-10)


def test_build_curve_accepts_path(settlements_path):
    curve = build_curve(settlements_path)
    assert curve.df(0) == pytest.approx(1.0, abs=1e-15)


def test_build_curve_reprices_contract_to_settlement(settlements_fixture, settlements_path):
    curve = build_curve(settlements_fixture)
    with open(settlements_path) as fh:
        for r in csv.DictReader(fh):
            di = DI1(r["ticker"], settlements_fixture.d0)
            assert di.price(curve) == pytest.approx(float(r["settlement_pu"]), abs=1e-6)


def test_fixture_rejects_non_increasing_du():
    with pytest.raises(ValueError):
        Fixture(date(2026, 6, 18), np.array([1, 5, 5]), np.array([0.14, 0.14, 0.14]))
