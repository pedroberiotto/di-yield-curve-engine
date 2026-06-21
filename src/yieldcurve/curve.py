from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np

from yieldcurve.daycount import DU_BASE, DateLike, business_days
from yieldcurve.interpolation import make_interpolator

_NumberOrDate = int | float | DateLike


class Curve:

    def __init__(
        self,
        d0: DateLike,
        dus: Sequence[int],
        dfs: Sequence[float],
        *,
        calendar: str = "ANBIMA",
        interpolation: str = "flat_forward",
    ) -> None:
        self.d0 = self._to_date(d0)
        self.calendar = calendar
        self.interpolation = interpolation

        dus_arr = np.asarray(dus, dtype=float)
        dfs_arr = np.asarray(dfs, dtype=float)
        if dus_arr.shape != dfs_arr.shape or dus_arr.ndim != 1:
            raise ValueError("dus and dfs must be 1-D and the same length")
        if dus_arr.size == 0:
            raise ValueError("the curve needs at least one node")
        if np.any(dus_arr <= 0):
            raise ValueError("node du must be > 0 (the du=0 node is implicit)")
        if np.any(dfs_arr <= 0) or np.any(dfs_arr > 1):
            raise ValueError("node DF must be in (0, 1]")
        order = np.argsort(dus_arr)
        dus_arr, dfs_arr = dus_arr[order], dfs_arr[order]
        if np.any(np.diff(dus_arr) <= 0):
            raise ValueError("node du must be strictly increasing")

        self.node_du = np.concatenate(([0.0], dus_arr))
        self.node_df = np.concatenate(([1.0], dfs_arr))
        self._logdf = np.log(self.node_df)
        self._interp = make_interpolator(interpolation, self.node_du, self._logdf)

    @classmethod
    def from_zeros(
        cls,
        d0: DateLike,
        dus: Sequence[int],
        zeros: Sequence[float],
        **kwargs,
    ) -> Curve:
        dus_arr = np.asarray(dus, dtype=float)
        zeros_arr = np.asarray(zeros, dtype=float)
        dfs = (1.0 + zeros_arr) ** (-dus_arr / DU_BASE)
        return cls(d0, dus, dfs, **kwargs)

    @staticmethod
    def _to_date(d: DateLike) -> date:
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            return date.fromisoformat(d)
        raise TypeError(f"invalid date: {d!r}")

    def du(self, t: _NumberOrDate) -> float:
        if isinstance(t, bool):
            raise TypeError("t cannot be bool")
        if isinstance(t, (int, float)):
            return float(t)
        return float(business_days(self.d0, t, self.calendar))

    def df(self, t: _NumberOrDate) -> float:
        du = self.du(t)
        if du < 0:
            raise ValueError("negative du is not supported")
        return float(np.exp(self._interp(du)))

    def zero(self, t: _NumberOrDate) -> float:
        du = self.du(t)
        if du <= 0:
            raise ValueError("zero undefined at du <= 0 (use the overnight anchor)")
        df = self.df(du)
        return df ** (-DU_BASE / du) - 1.0

    def forward(self, t1: _NumberOrDate, t2: _NumberOrDate) -> float:
        from yieldcurve.forward import forward_rate

        return forward_rate(self, t1, t2)

    def node_rates(self) -> tuple[np.ndarray, np.ndarray]:
        du = self.node_du[1:]
        df = self.node_df[1:]
        zero = df ** (-DU_BASE / du) - 1.0
        return du.copy(), zero

    def __repr__(self) -> str:
        return (
            f"Curve(d0={self.d0.isoformat()}, n_nodes={self.node_du.size - 1}, "
            f"interpolation={self.interpolation!r}, calendar={self.calendar!r})"
        )
