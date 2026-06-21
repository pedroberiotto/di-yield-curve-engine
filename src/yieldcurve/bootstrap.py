import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from yieldcurve.curve import Curve
from yieldcurve.instruments import rate_from_pu


@dataclass(frozen=True)
class Fixture:

    d0: date
    du: np.ndarray
    rate: np.ndarray

    def __post_init__(self) -> None:
        if self.du.shape != self.rate.shape:
            raise ValueError("du and rate must be the same length")
        if np.any(np.diff(self.du) <= 0):
            raise ValueError("fixture du must be strictly increasing")

    @property
    def max_du(self) -> int:
        return int(self.du[-1])


def load_settlements(path: str | Path, *, overnight_rate: float | None = None) -> Fixture:
    d0: date | None = None
    dus: list[int] = []
    spots: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if d0 is None:
                d0 = date.fromisoformat(row["refdate"][:10])
            du = int(row["business_days"])
            pu = float(row["settlement_pu"])
            dus.append(du)
            spots.append(rate_from_pu(pu, du))
    if d0 is None or not dus:
        raise ValueError(f"empty settlements: {path}")

    order = sorted(range(len(dus)), key=lambda i: dus[i])
    dus = [dus[i] for i in order]
    spots = [spots[i] for i in order]

    anchor = float(overnight_rate) if overnight_rate is not None else spots[0]
    if dus[0] != 1:
        dus.insert(0, 1)
        spots.insert(0, anchor)
    else:
        spots[0] = anchor
    return Fixture(d0, np.array(dus, dtype=int), np.array(spots, dtype=float))


def build_curve(
    source: str | Path | Fixture,
    *,
    interpolation: str = "flat_forward",
) -> Curve:
    fixture = source if isinstance(source, Fixture) else load_settlements(source)
    return Curve.from_zeros(
        fixture.d0, fixture.du, fixture.rate, interpolation=interpolation
    )
