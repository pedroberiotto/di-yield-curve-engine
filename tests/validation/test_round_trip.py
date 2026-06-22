import csv

import pytest

from yieldcurve.bootstrap import build_curve
from yieldcurve.instruments import DI1, NOTIONAL, pu_from_rate, rate_from_pu


def test_round_trip_settlements_recovers_settlement_pu(settlements_fixture, settlements_path):
    curve = build_curve(settlements_fixture)
    with open(settlements_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > 10
    max_err = 0.0
    for r in rows:
        di = DI1(r["ticker"], settlements_fixture.d0)
        max_err = max(max_err, abs(di.price(curve) - float(r["settlement_pu"])))
    assert max_err < 1e-6


def test_round_trip_all_nodes_recover_rate(settlements_fixture):
    curve = build_curve(settlements_fixture)
    for du, rate in zip(settlements_fixture.du, settlements_fixture.rate, strict=True):
        du = int(du)
        pu_out = NOTIONAL * curve.df(du)
        assert rate_from_pu(pu_out, du) == pytest.approx(rate, abs=1e-10)


def test_round_trip_pu_rate_reversible_at_every_node(settlements_fixture):
    for du, rate in zip(settlements_fixture.du, settlements_fixture.rate, strict=True):
        du = int(du)
        assert rate_from_pu(pu_from_rate(rate, du), du) == pytest.approx(rate, abs=1e-12)
