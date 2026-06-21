from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from yieldcurve.curve import Curve
from yieldcurve.daycount import business_days, decode_contract
from yieldcurve.instruments import (
    DI1,
    NOTIONAL,
    df_from_rate,
    pu_from_rate,
    rate_from_pu,
)

D0 = date(2026, 6, 19)


@pytest.mark.parametrize("rate", [0.0, 0.0525, 0.1415, 0.25])
@pytest.mark.parametrize("du", [1, 21, 252, 504, 1260])
def test_pu_rate_round_trip(rate, du):
    pu = pu_from_rate(rate, du)
    assert rate_from_pu(pu, du) == pytest.approx(rate, abs=1e-12)


def test_zero_rate_gives_par_pu():
    assert pu_from_rate(0.0, 252) == pytest.approx(NOTIONAL, abs=1e-9)


def test_df_consistent_with_pu():
    rate, du = 0.1415, 137
    assert df_from_rate(rate, du) == pytest.approx(pu_from_rate(rate, du) / NOTIONAL, abs=1e-15)


def test_pu_decreases_with_rate_and_term():
    assert pu_from_rate(0.10, 252) > pu_from_rate(0.15, 252)
    assert pu_from_rate(0.15, 252) > pu_from_rate(0.15, 504)


@pytest.mark.parametrize("fn", [pu_from_rate, df_from_rate])
def test_conversions_reject_nonpositive_du(fn):
    with pytest.raises(ValueError):
        fn(0.10, 0)


def test_rate_from_pu_rejects_bad_inputs():
    with pytest.raises(ValueError):
        rate_from_pu(0.0, 252)
    with pytest.raises(ValueError):
        rate_from_pu(NOTIONAL, 0)


def test_di1_maturity_and_du_match_daycount():
    di = DI1("F28", D0)
    assert di.maturity == decode_contract("F28")
    assert di.maturity == date(2028, 1, 3)
    assert di.du == business_days(D0, di.maturity)
    assert di.du == 386


def test_di1_normalizes_code():
    di = DI1("di1f28", D0)
    assert di.code == "F28"
    assert di.maturity == date(2028, 1, 3)


def test_di1_is_frozen():
    di = DI1("F28", D0)
    with pytest.raises(FrozenInstanceError):
        di.code = "F29"


def test_di1_pu_rate_inverse():
    di = DI1("F28", D0)
    pu = di.pu(0.14)
    assert di.rate(pu) == pytest.approx(0.14, abs=1e-12)
    assert di.df(0.14) == pytest.approx(pu / NOTIONAL, abs=1e-15)


def test_di1_pricing_against_curve_recovers_inputs():
    di = DI1("F28", D0)
    rate = 0.1375
    curve = Curve.from_zeros(D0, [di.du], [rate])
    assert di.price(curve) == pytest.approx(di.pu(rate), abs=1e-7)
    assert di.implied_rate(curve) == pytest.approx(rate, abs=1e-12)


def test_di1_rejects_matured_contract():
    di = DI1("F26", date(2026, 6, 19))
    with pytest.raises(ValueError):
        _ = di.du
