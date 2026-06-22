from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from yieldcurve.curve import Curve

BPS = 1e-4

CashFlow = tuple[int, float]


def cashflow_from_di1(di1, qty: float = 1.0) -> CashFlow:
    return (int(di1.du), qty * di1.notional)


def pv(curve: Curve, cashflows: Iterable[CashFlow]) -> float:
    return float(sum(amount * curve.df(int(du)) for du, amount in cashflows))


def _rebuild(curve: Curve, zeros: np.ndarray) -> Curve:
    du, _ = curve.node_rates()
    return Curve.from_zeros(
        curve.d0, du, zeros, calendar=curve.calendar, interpolation=curve.interpolation
    )


def shift_curve(curve: Curve, delta: float) -> Curve:
    _, zeros = curve.node_rates()
    return _rebuild(curve, zeros + delta)


def dv01(curve: Curve, cashflows: Iterable[CashFlow], bump: float = BPS) -> float:
    cfs = list(cashflows)
    base = pv(curve, cfs)
    up = pv(shift_curve(curve, bump), cfs)
    return base - up


@dataclass(frozen=True)
class KeyRateDV01:

    du: np.ndarray
    dv01: np.ndarray

    @property
    def total(self) -> float:
        return float(self.dv01.sum())


def key_rate_dv01(
    curve: Curve, cashflows: Iterable[CashFlow], bump: float = BPS
) -> KeyRateDV01:
    cfs = list(cashflows)
    du, zeros = curve.node_rates()
    base = pv(curve, cfs)
    out = np.empty(du.size)
    for i in range(du.size):
        bumped = zeros.copy()
        bumped[i] += bump
        out[i] = base - pv(_rebuild(curve, bumped), cfs)
    return KeyRateDV01(du.copy(), out)
