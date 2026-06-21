from datetime import date

import numpy as np
import pytest

from yieldcurve.curve import Curve
from yieldcurve.forward import (
    daily_forward,
    forward_curve,
    forward_factor,
    forward_rate,
)

D0 = date(2026, 6, 19)
DUS = [21, 63, 126, 252, 504]
ZEROS = [0.105, 0.108, 0.112, 0.118, 0.122]


@pytest.fixture
def curve():
    return Curve.from_zeros(D0, DUS, ZEROS)


def test_forward_factor_is_df_ratio(curve):
    assert forward_factor(curve, 63, 252) == pytest.approx(
        curve.df(63) / curve.df(252), abs=1e-15
    )


def test_forward_rate_chains_back_to_df(curve):
    du1, du2 = 63, 252
    f = forward_rate(curve, du1, du2)
    assert (1 + f) ** ((du2 - du1) / 252) == pytest.approx(
        curve.df(du1) / curve.df(du2), abs=1e-12
    )


def test_forward_rate_matches_curve_method(curve):
    assert forward_rate(curve, 100, 400) == pytest.approx(curve.forward(100, 400), abs=1e-15)


def test_forward_constant_within_segment(curve):
    f_seg = forward_rate(curve, 252, 504)
    assert forward_rate(curve, 300, 450) == pytest.approx(f_seg, abs=1e-12)
    assert forward_rate(curve, 252, 300) == pytest.approx(f_seg, abs=1e-12)


def test_flat_zero_curve_has_forward_equal_to_zero_rate():
    i = 0.13
    c = Curve.from_zeros(D0, DUS, [i] * len(DUS))
    for a, b in [(1, 252), (63, 504), (100, 101)]:
        assert forward_rate(c, a, b) == pytest.approx(i, abs=1e-12)


def test_daily_forward_matches_one_day_forward(curve):
    assert daily_forward(curve, 100) == pytest.approx(forward_rate(curve, 100, 101), abs=1e-15)


def test_forward_curve_consecutive(curve):
    terms = [21, 126, 252, 504]
    fc = forward_curve(curve, terms)
    assert fc.rate.size == len(terms) - 1
    assert np.array_equal(fc.du_start, np.array(terms[:-1], dtype=float))
    assert np.array_equal(fc.du_end, np.array(terms[1:], dtype=float))
    for i in range(len(terms) - 1):
        assert fc.rate[i] == pytest.approx(forward_rate(curve, terms[i], terms[i + 1]), abs=1e-15)


def test_forward_requires_ordered_terms(curve):
    with pytest.raises(ValueError):
        forward_rate(curve, 252, 252)
    with pytest.raises(ValueError):
        forward_factor(curve, 252, 100)
    with pytest.raises(ValueError):
        forward_curve(curve, [252, 100])
    with pytest.raises(ValueError):
        forward_curve(curve, [100])


def test_forward_accepts_dates(curve):
    d1, d2 = date(2026, 12, 1), date(2027, 6, 1)
    assert forward_rate(curve, d1, d2) == pytest.approx(
        forward_rate(curve, curve.du(d1), curve.du(d2)), abs=1e-15
    )
