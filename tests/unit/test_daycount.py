
from datetime import date

import pytest

from yieldcurve import daycount as dc


@pytest.mark.parametrize(
    "d, expected",
    [
        ("2025-01-02", True),
        ("2025-01-01", False),
        ("2025-01-04", False),
        ("2025-01-05", False),
        ("2025-12-25", False),
    ],
)
def test_is_business_day(d, expected):
    assert dc.is_business_day(d) is expected


def test_following_and_preceding():
    assert dc.following("2025-01-01") == date(2025, 1, 2)
    assert dc.preceding("2025-01-01") == date(2024, 12, 31)
    assert dc.following("2025-01-02") == date(2025, 1, 2)
    assert dc.preceding("2025-01-02") == date(2025, 1, 2)


def test_add_business_days():
    assert dc.add_business_days("2025-01-02", 1) == date(2025, 1, 3)
    assert dc.add_business_days("2025-01-03", -1) == date(2025, 1, 2)


@pytest.mark.parametrize(
    "d0, venc, expected",
    [
        ("2025-01-06", "2025-01-10", 4),
        ("2025-01-06", "2025-01-11", 5),
        ("2025-01-04", "2025-01-10", 4),
        ("2025-01-01", "2025-01-10", 6),
        ("2025-06-20", "2026-01-02", 137),
    ],
)
def test_business_days_values(d0, venc, expected):
    assert dc.business_days(d0, venc) == expected


def test_du_zero_when_same_or_inverted():
    assert dc.business_days("2025-01-06", "2025-01-06") == 0
    assert dc.business_days("2025-01-10", "2025-01-06") == 0


def test_du_one_to_next_business_day():
    d0 = "2025-01-06"
    nxt = dc.add_business_days(d0, 1)
    assert dc.business_days(d0, nxt) == 1


def test_du_half_open_excludes_venc():
    d0, mid, end = "2025-01-06", "2025-01-13", "2025-01-20"
    assert (
        dc.business_days(d0, mid) + dc.business_days(mid, end)
        == dc.business_days(d0, end)
    )


def test_du_additivity_over_partition():
    a, b, c = "2025-02-03", "2025-05-02", "2025-09-01"
    assert dc.business_days(a, b) + dc.business_days(b, c) == dc.business_days(a, c)


def test_year_fraction():
    du = dc.business_days("2025-06-20", "2026-01-02")
    assert du == 137
    assert dc.year_fraction("2025-06-20", "2026-01-02") == pytest.approx(137 / 252)


@pytest.mark.parametrize(
    "year, month, expected",
    [
        (2025, 1, date(2025, 1, 2)),
        (2025, 7, date(2025, 7, 1)),
        (2025, 9, date(2025, 9, 1)),
        (2025, 11, date(2025, 11, 3)),
        (2026, 1, date(2026, 1, 2)),
    ],
)
def test_first_business_day_of_month(year, month, expected):
    assert dc.first_business_day_of_month(year, month) == expected


def test_decode_contract_basic():
    assert dc.decode_contract("F26") == date(2026, 1, 2)
    assert dc.decode_contract("N25") == dc.first_business_day_of_month(2025, 7)


def test_decode_contract_accepts_prefix_and_case_and_4digit_year():
    assert dc.decode_contract("di1f26") == date(2026, 1, 2)
    assert dc.decode_contract("F2026") == date(2026, 1, 2)
    assert dc.decode_contract("  F26  ") == date(2026, 1, 2)


def test_decode_contract_all_months_map_to_correct_month():
    for letter, month in dc.MONTH_CODES.items():
        venc = dc.decode_contract(f"{letter}27")
        assert venc.year == 2027
        assert venc.month == month
        assert dc.is_business_day(venc)
        assert venc == dc.first_business_day_of_month(2027, month)


@pytest.mark.parametrize("bad", ["", "A26", "F2", "F260", "FF26", "26", "Fxx"])
def test_decode_contract_rejects_invalid(bad):
    with pytest.raises(ValueError):
        dc.decode_contract(bad)
