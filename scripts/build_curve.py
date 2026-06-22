import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yieldcurve.bootstrap import Fixture, build_curve, load_settlements
from yieldcurve.curve import Curve
from yieldcurve.instruments import NOTIONAL
from yieldcurve.risk import KeyRateDV01, key_rate_dv01

CURVES_DIR = ROOT / "data" / "curves"


def build(source: str | Path | Fixture) -> Curve:
    return build_curve(source)


def save_curve(curve: Curve, out_dir: Path = CURVES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    du, zero = curve.node_rates()
    path = out_dir / f"curve_{curve.d0.isoformat()}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["refdate", "business_days", "zero", "df"])
        for d, z in zip(du, zero, strict=True):
            w.writerow([curve.d0.isoformat(), int(d), f"{z:.10f}", f"{curve.df(int(d)):.12f}"])
    update_latest(out_dir)
    return path


def update_latest(out_dir: Path = CURVES_DIR) -> Path | None:
    files = sorted(out_dir.glob("curve_*.csv"))
    if not files:
        return None
    latest = out_dir / "latest.csv"
    shutil.copyfile(files[-1], latest)
    return latest


def reference_risk(curve: Curve) -> KeyRateDV01:
    du, _ = curve.node_rates()
    book = [(int(d), NOTIONAL) for d in du]
    return key_rate_dv01(curve, book)


def save_risk(kr: KeyRateDV01, d0: date, out_dir: Path = CURVES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"risk_{d0.isoformat()}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["refdate", "business_days", "dv01"])
        for d, v in zip(kr.du, kr.dv01, strict=True):
            w.writerow([d0.isoformat(), int(d), f"{float(v):.6f}"])
    return path


def benchmark(curve: Curve, du_list, rate_list) -> list[tuple[int, float, float, float]]:
    du_max = int(curve.node_du[-1])
    rows = []
    for du, anbima in zip(du_list, rate_list, strict=True):
        du = int(du)
        if du < 1 or du > du_max:
            continue
        di = curve.zero(du)
        rows.append((du, di, float(anbima), (di - anbima) * 1e4))
    return rows


def save_benchmark(rows, d0: date, out_dir: Path = CURVES_DIR, *, label: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{label}_" if label else ""
    path = out_dir / f"benchmark_{suffix}{d0.isoformat()}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["business_days", "rate_di", "rate_ref", "diff_bps"])
        for du, di, ref, bps in rows:
            w.writerow([du, f"{di:.10f}", f"{ref:.10f}", f"{bps:.3f}"])
    return path


def benchmark_summary(rows) -> dict[str, float]:
    diffs = [abs(r[3]) for r in rows]
    if not diffs:
        return {"n": 0, "mean_abs_bps": 0.0, "max_abs_bps": 0.0}
    return {
        "n": len(diffs),
        "mean_abs_bps": sum(diffs) / len(diffs),
        "max_abs_bps": max(diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline fetch->bootstrap->save + benchmark.")
    parser.add_argument("data", nargs="?", help="date YYYY-MM-DD (default: recent)")
    parser.add_argument("--settlements", help="use a local settlements CSV instead of downloading")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    from yieldcurve.daycount import preceding

    if args.settlements:
        fixture = load_settlements(args.settlements)
        refdate = fixture.d0
    else:
        import fetch_di1

        start = date.fromisoformat(args.data) if args.data else preceding(date.today())
        refdate, rows = fetch_di1.fetch_recent(start)
        fetch_di1.save(refdate, rows)
        overnight = fetch_di1.fetch_overnight(refdate)
        src = f"{overnight:.4%} (CDI/BCB)" if overnight else "front contract (BCB unavailable)"
        print(f"[build_curve] overnight = {src}")
        fixture = load_settlements(
            ROOT / "data" / "raw" / f"settlements_{refdate.isoformat()}.csv",
            overnight_rate=overnight,
        )

    curve = build(fixture)
    curve_path = save_curve(curve)
    print(f"[build_curve] d0={curve.d0.isoformat()}  nodes={curve.node_du.size - 1}")
    print(f"[build_curve] curve saved to {curve_path}")

    kr = reference_risk(curve)
    risk_path = save_risk(kr, curve.d0)
    print(f"[build_curve] risk saved to {risk_path}  (parallel DV01={kr.total:.2f})")

    if not args.no_benchmark:
        try:
            import fetch_anbima_ettj

            _, du, rate = fetch_anbima_ettj.fetch(refdate)
            rows = benchmark(curve, du, rate)
            bench_path = save_benchmark(rows, curve.d0)
            s = benchmark_summary(rows)
            print(
                f"[build_curve] ANBIMA benchmark: {s['n']} vertices  "
                f"|diff| mean={s['mean_abs_bps']:.2f} bps  max={s['max_abs_bps']:.2f} bps"
            )
            print(f"[build_curve] report saved to {bench_path}")
        except Exception as ex:
            print(f"[build_curve] ANBIMA benchmark unavailable: {type(ex).__name__}")

    try:
        import plot_curve

        print(f"[build_curve] plot saved to {plot_curve.plot()}")
    except Exception as ex:
        print(f"[build_curve] plot skipped: {type(ex).__name__}")


if __name__ == "__main__":
    main()
