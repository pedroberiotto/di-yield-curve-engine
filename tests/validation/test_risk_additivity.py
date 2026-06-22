import numpy as np
import pytest

from yieldcurve.bootstrap import build_curve
from yieldcurve.instruments import DI1
from yieldcurve.risk import cashflow_from_di1, dv01, key_rate_dv01


@pytest.fixture
def curve(settlements_fixture):
    return build_curve(settlements_fixture)


def test_additivity_on_real_portfolio(curve, settlements_fixture):
    positions = [("N26", 10), ("F28", -5), ("F30", 8), ("F33", -3)]
    cfs = [cashflow_from_di1(DI1(c, settlements_fixture.d0), q) for c, q in positions]
    par = dv01(curve, cfs)
    kr = key_rate_dv01(curve, cfs)
    assert abs(par) > 0
    assert kr.total == pytest.approx(par, rel=1e-6)


def test_keyrate_vector_covers_all_nodes(curve):
    cfs = [(int(curve.node_du[-1]), 1_000_000.0)]
    kr = key_rate_dv01(curve, cfs)
    du_nodes, _ = curve.node_rates()
    assert np.array_equal(kr.du, du_nodes)


def test_single_long_di1_has_positive_dv01(curve, settlements_fixture):
    cf = [cashflow_from_di1(DI1("F30", settlements_fixture.d0), 1)]
    assert dv01(curve, cf) > 0
