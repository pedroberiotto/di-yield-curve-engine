from datetime import date

import pytest

from yieldcurve.curve import Curve
from yieldcurve.daycount import business_days, decode_contract, get_calendar

ql = pytest.importorskip("QuantLib")


def _qd(d: date):
    return ql.Date(d.day, d.month, d.year)


def _anbima_ql_calendar():
    cal = ql.BespokeCalendar("ANBIMA")
    cal.addWeekend(ql.Saturday)
    cal.addWeekend(ql.Sunday)
    for h in get_calendar("ANBIMA").holidays:
        cal.addHoliday(_qd(h))
    return cal


D0 = date(2025, 6, 20)
CONTRACTS = ["N25", "V25", "F26", "J26", "N26", "F27", "F28", "F29", "F30"]
ZEROS = [0.1045, 0.1080, 0.1120, 0.1145, 0.1165, 0.1180, 0.1205, 0.1215, 0.1220]


@pytest.fixture(scope="module")
def setup():
    cal = _anbima_ql_calendar()
    dc = ql.Business252(cal)

    vencs = [decode_contract(c) for c in CONTRACTS]
    dus = [business_days(D0, v) for v in vencs]
    dfs = [(1 + z) ** (-du / 252) for du, z in zip(dus, ZEROS, strict=True)]

    curve = Curve.from_zeros(D0, dus, ZEROS)

    ql_dates = [_qd(D0)] + [_qd(v) for v in vencs]
    ql_dfs = [1.0] + dfs
    ql_curve = ql.DiscountCurve(ql_dates, ql_dfs, dc)
    ql_curve.enableExtrapolation()

    return {
        "cal": cal,
        "dc": dc,
        "vencs": vencs,
        "dus": dus,
        "curve": curve,
        "ql_curve": ql_curve,
    }


def test_du_matches_quantlib_business252(setup):
    dc = setup["dc"]
    for v, du in zip(setup["vencs"], setup["dus"], strict=True):
        ql_du = int(round(dc.dayCount(_qd(D0), _qd(v))))
        assert ql_du == du


def test_discount_factors_match_oracle_to_1e8(setup):
    curve, ql_curve = setup["curve"], setup["ql_curve"]
    last = setup["vencs"][-1]
    bizcal = get_calendar("ANBIMA")
    max_err = 0.0
    for d in bizcal.seq(D0, last):
        my = curve.df(d)
        ref = ql_curve.discount(_qd(d))
        max_err = max(max_err, abs(my - ref))
    assert max_err < 1e-8


def test_zero_rates_match_oracle_to_1e8(setup):
    curve, ql_curve = setup["curve"], setup["ql_curve"]
    dc = setup["dc"]
    bizcal = get_calendar("ANBIMA")
    vencs = setup["vencs"]
    max_err = 0.0
    for d in bizcal.seq(bizcal.following(date(2025, 6, 23)), vencs[-1]):
        my = curve.zero(d)
        ref = ql_curve.zeroRate(_qd(d), dc, ql.Compounded, ql.Annual).rate()
        max_err = max(max_err, abs(my - ref))
    assert max_err < 1e-8


def test_nodes_round_trip_exactly(setup):
    curve = setup["curve"]
    for du, z in zip(setup["dus"], ZEROS, strict=True):
        assert curve.zero(du) == pytest.approx(z, abs=1e-10)
