import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yieldcurve.daycount import is_business_day, preceding

RAW_DIR = ROOT / "data" / "raw"
MAX_LOOKBACK = 10


def _to_float(value) -> float:
    return float(str(value).strip().replace(",", "."))


def _to_du(value) -> int:
    return int(str(value).strip().replace(".", ""))


def _is_blank(value) -> bool:
    s = str(value).strip().lower()
    return s in ("", "nan", "none", "-")


def parse_pre_vertices(ettj_df) -> tuple[list[int], list[float]]:
    dus, rates = [], []
    for _, row in ettj_df.iterrows():
        if _is_blank(row["Prefixados"]):
            continue
        dus.append(_to_du(row["Vertice"]))
        rates.append(_to_float(row["Prefixados"]) / 100.0)
    order = sorted(range(len(dus)), key=lambda i: dus[i])
    return [dus[i] for i in order], [rates[i] for i in order]


def _previous_business_day(d: date) -> date:
    prev = d - timedelta(days=1)
    while not is_business_day(prev):
        prev -= timedelta(days=1)
    return prev


def fetch(data: str | date):
    import pyettj

    d = data.isoformat() if isinstance(data, date) else data
    d_br = "/".join(reversed(d.split("-"))) if "-" in d else d
    _params, ettj_df, _taxa, _erro = pyettj.get_ettj_anbima(d_br)
    if ettj_df is None or len(ettj_df) == 0:
        raise ValueError(f"no ANBIMA ETTJ for {d}")
    refdate = date.fromisoformat(d) if "-" in d else _br_to_iso(d)
    du, rate = parse_pre_vertices(ettj_df)
    return refdate, du, rate


def fetch_recent(start: date):
    d = start
    for _ in range(MAX_LOOKBACK):
        try:
            return fetch(d)
        except Exception as ex:
            print(f"[fetch_anbima] {d.isoformat()}: {type(ex).__name__}; going back…")
            d = _previous_business_day(d)
    raise SystemExit(f"no ANBIMA ETTJ in the last {MAX_LOOKBACK} business days")


def _br_to_iso(d_br: str) -> date:
    dd, mm, yyyy = d_br.split("/")
    return date(int(yyyy), int(mm), int(dd))


def save(refdate: date, du, rate, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"anbima_pre_{refdate.isoformat()}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["refdate", "business_days", "rate"])
        for d, r in zip(du, rate, strict=True):
            w.writerow([refdate.isoformat(), d, f"{r:.6f}"])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ANBIMA prefixed ETTJ.")
    parser.add_argument("data", nargs="?", help="date YYYY-MM-DD (default: recent)")
    args = parser.parse_args()

    start = date.fromisoformat(args.data) if args.data else preceding(date.today())
    refdate, du, rate = fetch_recent(start)
    path = save(refdate, du, rate)
    print(
        f"[fetch_anbima] d0={refdate.isoformat()}  vertices={len(du)}  "
        f"du={du[0]}..{du[-1]}  rate={min(rate):.4%}..{max(rate):.4%}"
    )
    print(f"[fetch_anbima] saved to {path}")


if __name__ == "__main__":
    main()
