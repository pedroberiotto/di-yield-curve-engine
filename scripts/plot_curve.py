import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVES_DIR = ROOT / "data" / "curves"
FIG_DIR = ROOT / "reports" / "figures"


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_latest():
    latest = CURVES_DIR / "latest.csv"
    rows = _read_csv(latest)
    d0 = date.fromisoformat(rows[0]["refdate"])
    du = [int(r["business_days"]) for r in rows]
    zero = [float(r["zero"]) for r in rows]
    return d0, du, zero


def load_benchmark(d0):
    path = CURVES_DIR / f"benchmark_{d0.isoformat()}.csv"
    if not path.exists():
        return None
    rows = _read_csv(path)
    return {
        "du": [int(r["business_days"]) for r in rows],
        "di": [float(r["rate_di"]) for r in rows],
        "ref": [float(r["rate_ref"]) for r in rows],
        "bps": [float(r["diff_bps"]) for r in rows],
    }


def plot(out: Path = FIG_DIR / "curve_latest.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d0, du, zero = load_latest()
    bench = load_benchmark(d0)

    years = [d / 252 for d in du]
    zero_pct = [z * 100 for z in zero]

    if bench:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(9.5, 6.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        axr = None

    ax.plot(years, zero_pct, "-", color="#1f4e79", lw=1.8, label="DI bootstrap (our engine)")
    ax.plot(years[1:], zero_pct[1:], ".", color="#1f4e79", ms=5)
    if bench:
        by = [b / 252 for b in bench["du"]]
        ax.plot(by, [r * 100 for r in bench["ref"]], "s", color="#c0504d", ms=5,
                label="ANBIMA prefixed (LTN/NTN-F)")

    ax.set_ylabel("Zero rate (% p.a., base 252)")
    ax.set_title(f"DI yield curve — {d0.isoformat()}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    if axr is not None:
        absbps = [abs(b) for b in bench["bps"]]
        mean_bps = sum(absbps) / len(absbps)
        max_bps = max(absbps)
        axr.axhline(0, color="0.6", lw=0.8)
        axr.bar(by, bench["bps"], width=0.18, color="#7f7f7f")
        axr.set_ylabel("DI − ANBIMA (bps)")
        axr.set_xlabel("Maturity (years = du / 252)")
        axr.grid(True, alpha=0.3)
        axr.text(0.99, 0.06, f"|diff| mean {mean_bps:.1f} bps · max {max_bps:.1f} bps",
                 transform=axr.transAxes, ha="right", va="bottom", fontsize=9, color="0.3")
    else:
        ax.set_xlabel("Maturity (years = du / 252)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    path = plot()
    print(f"[plot_curve] saved {path}")


if __name__ == "__main__":
    sys.exit(main())
