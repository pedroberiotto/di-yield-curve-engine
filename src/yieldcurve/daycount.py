

from datetime import date, datetime
from functools import cache

from bizdays import Calendar

DU_BASE = 252

MONTH_CODES: dict[str, int] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

_YEAR_PIVOT = 2000

DateLike = date | datetime | str


@cache
def get_calendar(name: str = "ANBIMA") -> Calendar:
    return Calendar.load(name)


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(d)
    raise TypeError(f"invalid date: {d!r} ({type(d).__name__})")


def is_business_day(d: DateLike, calendar: str = "ANBIMA") -> bool:
    return get_calendar(calendar).isbizday(_to_date(d))


def following(d: DateLike, calendar: str = "ANBIMA") -> date:
    return get_calendar(calendar).following(_to_date(d))


def preceding(d: DateLike, calendar: str = "ANBIMA") -> date:
    return get_calendar(calendar).preceding(_to_date(d))


def add_business_days(d: DateLike, n: int, calendar: str = "ANBIMA") -> date:
    return get_calendar(calendar).offset(_to_date(d), n)


def business_days(d0: DateLike, venc: DateLike, calendar: str = "ANBIMA") -> int:
    cal = get_calendar(calendar)
    start = _to_date(d0)
    end = _to_date(venc)
    if start >= end:
        return 0
    return cal.bizdays(cal.following(start), cal.following(end))


def year_fraction(d0: DateLike, venc: DateLike, calendar: str = "ANBIMA") -> float:
    return business_days(d0, venc, calendar) / DU_BASE


def first_business_day_of_month(year: int, month: int, calendar: str = "ANBIMA") -> date:
    return get_calendar(calendar).following(date(year, month, 1))


def decode_contract(code: str, calendar: str = "ANBIMA") -> date:
    raw = code.strip().upper()
    if raw.startswith("DI1"):
        raw = raw[3:]
    if not raw:
        raise ValueError(f"empty contract code: {code!r}")

    letter, year_part = raw[0], raw[1:]
    if letter not in MONTH_CODES:
        raise ValueError(f"invalid month letter {letter!r} in code {code!r}")
    if not year_part.isdigit() or len(year_part) not in (2, 4):
        raise ValueError(f"invalid year {year_part!r} in code {code!r}")

    year = int(year_part)
    if len(year_part) == 2:
        year += _YEAR_PIVOT
    month = MONTH_CODES[letter]
    return first_business_day_of_month(year, month, calendar)
