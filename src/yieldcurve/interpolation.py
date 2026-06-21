from typing import Protocol

import numpy as np


class Interpolator(Protocol):

    def __call__(self, du: float | np.ndarray) -> float | np.ndarray: ...


def _as_array(du: float | np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.atleast_1d(np.asarray(du, dtype=float))
    return arr, np.isscalar(du) or np.ndim(du) == 0


class PiecewiseLogLinear:

    def __init__(self, dus: np.ndarray, logdfs: np.ndarray) -> None:
        self.x = np.asarray(dus, dtype=float)
        self.y = np.asarray(logdfs, dtype=float)
        if self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("interpolation needs at least 2 nodes")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("nodes (du) must be strictly increasing")

    def __call__(self, du: float | np.ndarray) -> float | np.ndarray:
        x, scalar = _as_array(du)
        idx = np.searchsorted(self.x, x, side="right") - 1
        idx = np.clip(idx, 0, self.x.size - 2)
        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]
        t = (x - x0) / (x1 - x0)
        out = y0 + t * (y1 - y0)
        return float(out[0]) if scalar else out


class CubicLogDF:

    def __init__(self, dus: np.ndarray, logdfs: np.ndarray) -> None:
        self.x = np.asarray(dus, dtype=float)
        self.y = np.asarray(logdfs, dtype=float)
        if self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("interpolation needs at least 2 nodes")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("nodes (du) must be strictly increasing")
        self._m = self._second_derivatives()

    def _second_derivatives(self) -> np.ndarray:
        x, y = self.x, self.y
        n = x.size
        m = np.zeros(n)
        if n < 3:
            return m
        h = np.diff(x)
        a = h[:-1]
        b = 2.0 * (h[:-1] + h[1:])
        c = h[1:]
        d = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        m[1:-1] = self._solve_tridiagonal(a, b, c, d)
        return m

    @staticmethod
    def _solve_tridiagonal(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        n = b.size
        cp = np.empty(n)
        dp = np.empty(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / denom if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        out = np.empty(n)
        out[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            out[i] = dp[i] - cp[i] * out[i + 1]
        return out

    def _segment_slope(self, i: int, at_left: bool) -> float:
        h = self.x[i + 1] - self.x[i]
        dy = (self.y[i + 1] - self.y[i]) / h
        m0, m1 = self._m[i], self._m[i + 1]
        if at_left:
            return dy - h * (2.0 * m0 + m1) / 6.0
        return dy + h * (m0 + 2.0 * m1) / 6.0

    def __call__(self, du: float | np.ndarray) -> float | np.ndarray:
        x, scalar = _as_array(du)
        out = np.empty_like(x)
        lo, hi = self.x[0], self.x[-1]

        below = x < lo
        above = x > hi
        inside = ~(below | above)

        if np.any(inside):
            xi = x[inside]
            idx = np.searchsorted(self.x, xi, side="right") - 1
            idx = np.clip(idx, 0, self.x.size - 2)
            h = self.x[idx + 1] - self.x[idx]
            A = (self.x[idx + 1] - xi) / h
            B = (xi - self.x[idx]) / h
            m0, m1 = self._m[idx], self._m[idx + 1]
            out[inside] = (
                A * self.y[idx]
                + B * self.y[idx + 1]
                + ((A**3 - A) * m0 + (B**3 - B) * m1) * h**2 / 6.0
            )
        if np.any(below):
            slope = self._segment_slope(0, at_left=True)
            out[below] = self.y[0] + slope * (x[below] - lo)
        if np.any(above):
            slope = self._segment_slope(self.x.size - 2, at_left=False)
            out[above] = self.y[-1] + slope * (x[above] - hi)
        return float(out[0]) if scalar else out


INTERPOLATORS: dict[str, type] = {
    "flat_forward": PiecewiseLogLinear,
    "log_linear_df": PiecewiseLogLinear,
    "cubic": CubicLogDF,
}


def make_interpolator(
    name: str, dus: np.ndarray, logdfs: np.ndarray
) -> Interpolator:
    try:
        cls = INTERPOLATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown interpolation {name!r}; options: {sorted(INTERPOLATORS)}"
        ) from None
    return cls(dus, logdfs)
