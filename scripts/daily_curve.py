import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_curve
import fetch_di1

from yieldcurve.bootstrap import Fixture, load_settlements
from yieldcurve.daycount import is_business_day

CURVES_DIR = ROOT / "data" / "curves"
RAW_DIR = ROOT / "data" / "raw"

N_BACKFILL = 5
MIN_VERTICES = 20


def recent_business_days(today: date, n: int) -> list[date]:
    out: list[date] = []
    d = today
    while len(out) < n:
        if is_business_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return out


def curve_exists(d0: date, curves_dir: Path = CURVES_DIR) -> bool:
    return (curves_dir / f"curve_{d0.isoformat()}.csv").exists()


def benchmark_exists(d0: date, curves_dir: Path = CURVES_DIR) -> bool:
    return (curves_dir / f"benchmark_{d0.isoformat()}.csv").exists()


def pending_days(
    today: date,
    n: int = N_BACKFILL,
    curves_dir: Path = CURVES_DIR,
    *,
    with_benchmark: bool = True,
) -> list[date]:
    days = recent_business_days(today, n)

    def pending(d: date) -> bool:
        if not curve_exists(d, curves_dir):
            return True
        return with_benchmark and not benchmark_exists(d, curves_dir)

    return sorted(d for d in days if pending(d))


def check_refdate(refdate: date, target: date) -> None:
    if refdate != target:
        raise ValueError(f"refdate {refdate.isoformat()} != target {target.isoformat()}")


def sanity_check(fixture: Fixture, min_vertices: int = MIN_VERTICES) -> None:
    n = fixture.du.size
    if n < min_vertices:
        raise ValueError(f"too few vertices: {n} < {min_vertices}")
    if int(fixture.du[0]) != 1:
        raise ValueError("missing overnight anchor (du=1)")
    if np.any(np.isnan(fixture.rate)):
        raise ValueError("rate has NaN")
    if np.any((fixture.rate <= 0) | (fixture.rate >= 1)):
        raise ValueError("rate outside (0, 1)")


def process_day(
    target: date,
    *,
    curves_dir: Path = CURVES_DIR,
    raw_dir: Path = RAW_DIR,
    min_vertices: int = MIN_VERTICES,
    with_benchmark: bool = True,
) -> str:
    curve_path = curves_dir / f"curve_{target.isoformat()}.csv"
    settlements_path = raw_dir / f"settlements_{target.isoformat()}.csv"
    if curve_path.exists() and settlements_path.exists():
        if with_benchmark and not benchmark_exists(target, curves_dir):
            curve = build_curve.build(load_settlements(settlements_path))
            ok = _best_effort_benchmark(curve, curves_dir)
            return "benchmark-ok" if ok else "benchmark-unavailable"
        return "already-complete"

    try:
        text = fetch_di1.fetch(target)
        refdate, rows = fetch_di1.parse_trade_information(text)
    except Exception as exc:
        return f"no-data ({type(exc).__name__})"

    if refdate is None or not rows:
        return "no-data (empty)"
    try:
        check_refdate(refdate, target)
    except ValueError as exc:
        return f"not-published ({exc})"

    fetch_di1.save(refdate, rows, raw_dir)
    overnight = fetch_di1.fetch_overnight(refdate)
    fixture = load_settlements(
        raw_dir / f"settlements_{refdate.isoformat()}.csv", overnight_rate=overnight
    )
    try:
        sanity_check(fixture, min_vertices)
    except ValueError as exc:
        return f"sanity-failed ({exc})"

    curve = build_curve.build(fixture)
    build_curve.save_curve(curve, curves_dir)
    build_curve.save_risk(build_curve.reference_risk(curve), curve.d0, curves_dir)

    if with_benchmark:
        _best_effort_benchmark(curve, curves_dir)
    return "ok"


def _best_effort_benchmark(curve, curves_dir: Path) -> bool:
    try:
        import fetch_anbima_ettj

        _, du, rate = fetch_anbima_ettj.fetch(curve.d0)
        rows = build_curve.benchmark(curve, du, rate)
        build_curve.save_benchmark(rows, curve.d0, curves_dir)
        return True
    except Exception as exc:
        print(f"  ANBIMA benchmark unavailable: {type(exc).__name__}")
        return False


def _best_effort_plot() -> None:
    try:
        import plot_curve

        plot_curve.plot()
    except Exception as exc:
        print(f"  plot skipped: {type(exc).__name__}")


def run(
    today: date | None = None,
    *,
    n: int = N_BACKFILL,
    curves_dir: Path = CURVES_DIR,
    raw_dir: Path = RAW_DIR,
    min_vertices: int = MIN_VERTICES,
    with_benchmark: bool = True,
) -> list[tuple[date, str]]:
    today = today or date.today()
    targets = pending_days(today, n, curves_dir, with_benchmark=with_benchmark)
    if not targets:
        print(f"[daily] nothing pending in the last {n} business days up to {today.isoformat()}")
        return []

    results: list[tuple[date, str]] = []
    for t in targets:
        status = process_day(
            t,
            curves_dir=curves_dir,
            raw_dir=raw_dir,
            min_vertices=min_vertices,
            with_benchmark=with_benchmark,
        )
        print(f"[daily] {t.isoformat()}: {status}")
        results.append((t, status))
    built = sum(1 for _, s in results if s == "ok")
    print(f"[daily] {built}/{len(results)} new curve(s) built")
    _best_effort_plot()
    return results


def main() -> None:
    run()


if __name__ == "__main__":
    main()
