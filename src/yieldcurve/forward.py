from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from yieldcurve.daycount import DU_BASE

if TYPE_CHECKING:
    from yieldcurve.curve import Curve


def forward_factor(curve: Curve, t1, t2) -> float:
    du1, du2 = curve.du(t1), curve.du(t2)
    if du2 <= du1:
        raise ValueError("forward requires du(t2) > du(t1)")
    return curve.df(du1) / curve.df(du2)


def forward_rate(curve: Curve, t1, t2) -> float:
    du1, du2 = curve.du(t1), curve.du(t2)
    if du2 <= du1:
        raise ValueError("forward requires du(t2) > du(t1)")
    return forward_factor(curve, du1, du2) ** (DU_BASE / (du2 - du1)) - 1.0


def daily_forward(curve: Curve, du: int) -> float:
    return forward_rate(curve, du, du + 1)


@dataclass(frozen=True)
class ForwardCurve:

    du_start: np.ndarray
    du_end: np.ndarray
    rate: np.ndarray


def forward_curve(curve: Curve, terms) -> ForwardCurve:
    dus = np.array([curve.du(t) for t in terms], dtype=float)
    if dus.size < 2:
        raise ValueError("forward_curve needs at least 2 terms")
    if np.any(np.diff(dus) <= 0):
        raise ValueError("terms must be strictly increasing")
    rates = np.array(
        [forward_rate(curve, dus[i], dus[i + 1]) for i in range(dus.size - 1)]
    )
    return ForwardCurve(dus[:-1], dus[1:], rates)
