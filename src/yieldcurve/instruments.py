from dataclasses import dataclass
from datetime import date, datetime
from functools import cached_property

from yieldcurve.daycount import (
    DU_BASE,
    DateLike,
    business_days,
    decode_contract,
)

NOTIONAL = 100_000.0


def pu_from_rate(rate: float, du: int, notional: float = NOTIONAL) -> float:
    if du <= 0:
        raise ValueError("du must be > 0")
    return notional / (1.0 + rate) ** (du / DU_BASE)


def rate_from_pu(pu: float, du: int, notional: float = NOTIONAL) -> float:
    if du <= 0:
        raise ValueError("du must be > 0")
    if pu <= 0:
        raise ValueError("PU must be > 0")
    return (notional / pu) ** (DU_BASE / du) - 1.0


def df_from_rate(rate: float, du: int) -> float:
    if du <= 0:
        raise ValueError("du must be > 0")
    return (1.0 + rate) ** (-du / DU_BASE)


@dataclass(frozen=True)
class DI1:

    code: str
    d0: date
    calendar: str = "ANBIMA"
    notional: float = NOTIONAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "d0", _to_date(self.d0))
        code = self.code.strip().upper()
        if code.startswith("DI1"):
            code = code[3:]
        object.__setattr__(self, "code", code)

    @cached_property
    def maturity(self) -> date:
        return decode_contract(self.code, self.calendar)

    @cached_property
    def du(self) -> int:
        du = business_days(self.d0, self.maturity, self.calendar)
        if du <= 0:
            raise ValueError(
                f"contract {self.code} matures {self.maturity} <= d0 {self.d0}"
            )
        return du

    def pu(self, rate: float) -> float:
        return pu_from_rate(rate, self.du, self.notional)

    def rate(self, pu: float) -> float:
        return rate_from_pu(pu, self.du, self.notional)

    def df(self, rate: float) -> float:
        return df_from_rate(rate, self.du)

    def price(self, curve) -> float:
        return self.notional * curve.df(self.du)

    def implied_rate(self, curve) -> float:
        return curve.zero(self.du)


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(d)
    raise TypeError(f"invalid date: {d!r}")
