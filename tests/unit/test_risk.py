from datetime import date

import numpy as np
import pytest

from yieldcurve.curve import Curve
from yieldcurve.instruments import DI1, NOTIONAL
from yieldcurve.risk import (
    cashflow_from_di1,
    dv01,
    key_rate_dv01,
    pv,
    shift_curve,
)

D0 = date(2026, 6, 19)
DUS = [21, 63, 126, 252, 504, 756]
ZEROS = [0.105, 0.108, 0.112, 0.118, 0.122, 0.124]


@pytest.fixture
def curve():
    return Curve.from_zeros(D0, DUS, ZEROS)


def test_pv_single_cashflow(curve):
    assert pv(curve, [(252, 1000.0)]) == pytest.approx(1000.0 * curve.df(252), abs=1e-9)


def test_pv_is_additive_over_cashflows(curve):
    cfs = [(126, 500.0), (504, -200.0)]
    assert pv(curve, cfs) == pytest.approx(
        pv(curve, [cfs[0]]) + pv(curve, [cfs[1]]), abs=1e-9
    )


def test_shift_zero_is_identity(curve):
    c0 = shift_curve(curve, 0.0)
    for du in (10, 100, 300, 600):
        assert c0.df(du) == pytest.approx(curve.df(du), abs=1e-14)


def test_shift_up_lowers_pv(curve):
    cfs = [(252, 1.0)]
    assert pv(shift_curve(curve, 1e-4), cfs) < pv(curve, cfs)


def test_dv01_positive_for_long_position(curve):
    assert dv01(curve, [(504, 1_000_000.0)]) > 0


def test_dv01_scales_with_position(curve):
    d1 = dv01(curve, [(504, 1.0)])
    d10 = dv01(curve, [(504, 10.0)])
    assert d10 == pytest.approx(10 * d1, abs=1e-12)


def test_key_rate_aligned_with_nodes(curve):
    kr = key_rate_dv01(curve, [(252, 1_000.0)])
    du_nodes, _ = curve.node_rates()
    assert np.array_equal(kr.du, du_nodes)
    assert kr.dv01.size == du_nodes.size


def test_key_rate_localized_to_node_for_cashflow_on_node(curve):
    kr = key_rate_dv01(curve, [(252, 1_000_000.0)])
    idx = list(kr.du).index(252.0)
    mask = np.ones(kr.du.size, dtype=bool)
    mask[idx] = False
    assert abs(kr.dv01[idx]) > 0
    assert np.allclose(kr.dv01[mask], 0.0, atol=1e-9)


def test_key_rate_only_bracketing_nodes_for_interior_cashflow(curve):
    kr = key_rate_dv01(curve, [(378, 1_000_000.0)])
    nz = {int(d) for d, v in zip(kr.du, kr.dv01, strict=True) if abs(v) > 1e-9}
    assert nz == {252, 504}


def test_additivity_sum_keyrate_equals_parallel(curve):
    cfs = [(126, 1_000_000.0), (504, -400_000.0), (756, 250_000.0)]
    par = dv01(curve, cfs)
    kr = key_rate_dv01(curve, cfs)
    assert kr.total == pytest.approx(par, rel=1e-6)


def test_cashflow_from_di1():
    di = DI1("F28", D0)
    du, amt = cashflow_from_di1(di, qty=3)
    assert du == di.du
    assert amt == pytest.approx(3 * NOTIONAL, abs=1e-9)
