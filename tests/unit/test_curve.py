from datetime import date

import numpy as np
import pytest

from yieldcurve.curve import Curve
from yieldcurve.daycount import business_days

D0 = date(2025, 6, 20)
DUS = [21, 126, 252, 504, 756]
ZEROS = [0.105, 0.110, 0.115, 0.120, 0.122]


@pytest.fixture
def curve():
    return Curve.from_zeros(D0, DUS, ZEROS)


def test_df_at_zero_is_one(curve):
    assert curve.df(0) == pytest.approx(1.0, abs=1e-15)


def test_passes_exactly_through_nodes(curve):
    for du, z in zip(DUS, ZEROS, strict=True):
        df_node = (1 + z) ** (-du / 252)
        assert curve.df(du) == pytest.approx(df_node, abs=1e-12)
        assert curve.zero(du) == pytest.approx(z, abs=1e-12)


def test_df_monotonic_decreasing_under_positive_rates(curve):
    grid = np.arange(0, 760, 1.0)
    dfs = np.array([curve.df(d) for d in grid])
    assert np.all(np.diff(dfs) < 0)
    assert dfs[0] == pytest.approx(1.0, abs=1e-15)


def test_zero_undefined_at_du_zero(curve):
    with pytest.raises(ValueError):
        curve.zero(0)


def test_forward_constant_within_segment(curve):
    a, b = 252, 504
    f_full = curve.forward(a, b)
    f_sub1 = curve.forward(a, a + 50)
    f_sub2 = curve.forward(b - 50, b)
    assert f_sub1 == pytest.approx(f_full, abs=1e-12)
    assert f_sub2 == pytest.approx(f_full, abs=1e-12)


def test_forward_chains_back_to_df(curve):
    du1, du2 = 126, 504
    f = curve.forward(du1, du2)
    lhs = (1 + f) ** ((du2 - du1) / 252)
    assert lhs == pytest.approx(curve.df(du1) / curve.df(du2), abs=1e-12)


def test_forward_requires_ordered_terms(curve):
    with pytest.raises(ValueError):
        curve.forward(252, 252)
    with pytest.raises(ValueError):
        curve.forward(504, 252)


def test_date_maps_to_du(curve):
    venc = date(2026, 1, 2)
    du = business_days(D0, venc)
    assert curve.du(venc) == du
    assert curve.df(venc) == pytest.approx(curve.df(du), abs=1e-15)
    assert curve.zero(venc) == pytest.approx(curve.zero(du), abs=1e-15)


def test_date_string_accepted(curve):
    assert curve.df("2026-01-02") == pytest.approx(curve.df(date(2026, 1, 2)), abs=1e-15)


def test_from_zeros_matches_direct_df_construction():
    dfs = [(1 + z) ** (-du / 252) for du, z in zip(DUS, ZEROS, strict=True)]
    c1 = Curve.from_zeros(D0, DUS, ZEROS)
    c2 = Curve(D0, DUS, dfs)
    for du in (10, 100, 300, 600):
        assert c1.df(du) == pytest.approx(c2.df(du), abs=1e-15)


def test_cubic_also_passes_through_nodes():
    c = Curve.from_zeros(D0, DUS, ZEROS, interpolation="cubic")
    for du, z in zip(DUS, ZEROS, strict=True):
        assert c.zero(du) == pytest.approx(z, abs=1e-10)


def test_unsorted_nodes_are_sorted():
    c = Curve.from_zeros(D0, [252, 21, 126], [0.115, 0.105, 0.110])
    assert list(c.node_du) == [0.0, 21.0, 126.0, 252.0]


@pytest.mark.parametrize(
    "dus, dfs",
    [
        ([0, 252], [1.0, 0.9]),
        ([252, 252], [0.9, 0.8]),
        ([252], [1.5]),
        ([252], [-0.1]),
    ],
)
def test_invalid_nodes_rejected(dus, dfs):
    with pytest.raises(ValueError):
        Curve(D0, dus, dfs)


def test_bool_term_rejected(curve):
    with pytest.raises(TypeError):
        curve.df(True)
