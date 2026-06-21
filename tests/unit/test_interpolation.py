import math

import numpy as np
import pytest

from yieldcurve.interpolation import (
    CubicLogDF,
    PiecewiseLogLinear,
    make_interpolator,
)

DUS = np.array([0.0, 21.0, 126.0, 252.0, 504.0])
LOGDFS = np.array([0.0, -0.008, -0.05, -0.10, -0.21])


def test_passes_through_nodes():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    for x, y in zip(DUS, LOGDFS, strict=True):
        assert interp(x) == pytest.approx(y, abs=1e-12)


def test_flat_forward_is_linear_in_logdf():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    mid = (DUS[1] + DUS[2]) / 2
    assert interp(mid) == pytest.approx((LOGDFS[1] + LOGDFS[2]) / 2, abs=1e-12)


def test_flat_forward_constant_forward_between_nodes():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    a, b = DUS[2], DUS[3]
    f_seg = -(LOGDFS[3] - LOGDFS[2]) / (b - a)
    for du in np.linspace(a, b, 7)[:-1]:
        f_local = -(interp(du + 1) - interp(du))
        assert f_local == pytest.approx(f_seg, abs=1e-12)


def test_flat_forward_vectorized_matches_scalar():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    xs = np.array([10.0, 60.0, 300.0])
    vec = interp(xs)
    assert isinstance(vec, np.ndarray)
    for x, v in zip(xs, vec, strict=True):
        assert interp(x) == pytest.approx(v, abs=1e-14)


def test_flat_forward_linear_extrapolation_beyond_last_node():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    slope = (LOGDFS[-1] - LOGDFS[-2]) / (DUS[-1] - DUS[-2])
    far = DUS[-1] + 100.0
    assert interp(far) == pytest.approx(LOGDFS[-1] + slope * 100.0, abs=1e-12)


def test_scalar_returns_float():
    interp = PiecewiseLogLinear(DUS, LOGDFS)
    assert isinstance(interp(50.0), float)


def test_cubic_passes_through_nodes():
    interp = CubicLogDF(DUS, LOGDFS)
    for x, y in zip(DUS, LOGDFS, strict=True):
        assert interp(x) == pytest.approx(y, abs=1e-10)


def test_cubic_natural_bc_second_derivative_zero_at_ends():
    interp = CubicLogDF(DUS, LOGDFS)
    assert interp._m[0] == pytest.approx(0.0, abs=1e-12)
    assert interp._m[-1] == pytest.approx(0.0, abs=1e-12)


def test_cubic_smoother_than_linear_but_close_at_dense_nodes():
    x = np.array([0.0, 252.0])
    y = np.array([0.0, -0.1])
    cub = CubicLogDF(x, y)
    lin = PiecewiseLogLinear(x, y)
    assert cub(126.0) == pytest.approx(lin(126.0), abs=1e-12)


def test_registry_aliases_flat_forward_and_log_linear():
    a = make_interpolator("flat_forward", DUS, LOGDFS)
    b = make_interpolator("log_linear_df", DUS, LOGDFS)
    assert type(a) is type(b) is PiecewiseLogLinear
    assert a(77.0) == pytest.approx(b(77.0), abs=1e-15)


def test_registry_rejects_unknown():
    with pytest.raises(ValueError):
        make_interpolator("spline_magico", DUS, LOGDFS)


@pytest.mark.parametrize("cls", [PiecewiseLogLinear, CubicLogDF])
def test_requires_increasing_nodes(cls):
    with pytest.raises(ValueError):
        cls(np.array([0.0, 100.0, 50.0]), np.array([0.0, -0.05, -0.02]))


def test_log_linear_recovers_known_forward():
    i = 0.12
    f_daily = (1 + i) ** (1 / 252) - 1
    dus = np.array([0.0, 252.0])
    logdfs = np.array([0.0, math.log((1 + i) ** -1)])
    interp = PiecewiseLogLinear(dus, logdfs)
    df1 = math.exp(interp(1.0))
    assert df1 == pytest.approx((1 + i) ** (-1 / 252), abs=1e-14)
    assert (1 / df1 - 1) == pytest.approx(f_daily, abs=1e-14)
