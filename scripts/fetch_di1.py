import argparse
import csv
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yieldcurve.daycount import is_business_day, preceding
from yieldcurve.instruments import DI1, rate_from_pu

RAW_DIR = ROOT / "data" / "raw"
MAX_LOOKBACK = 10
_B3_API = "https://arquivos.b3.com.br/api/download"
_TICKER_RE = re.compile(r"^DI1[FGHJKMNQUVXZ]\d{2}$")
_BCB_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados"


class NoSettlementError(RuntimeError):
    pass


def _to_float(value: str) -> float | None:
    v = value.strip()
    if v in ("", "-"):
        return None
    return float(v.replace(".", "").replace(",", "."))


def parse_bcb_overnight(payload) -> float | None:
    if not payload:
        return None
    return float(str(payload[-1]["valor"]).replace(",", ".")) / 100.0


def fetch_overnight(data: str | date) -> float | None:
    import requests

    d = data.strftime("%d/%m/%Y") if isinstance(data, date) else data
    try:
        r = requests.get(
            _BCB_SGS,
            params={"formato": "json", "dataInicial": d, "dataFinal": d},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        if r.status_code != 200:
            return None
        return parse_bcb_overnight(r.json())
    except Exception:
        return None


def parse_trade_information(text: str) -> tuple[date | None, list[tuple[str, float, float]]]:
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("RptDt;")), None)
    if header_idx is None:
        raise ValueError("header 'RptDt;...' not found in file")
    cols = lines[header_idx].split(";")
    ix = {c: i for i, c in enumerate(cols)}

    refdate: date | None = None
    rows: list[tuple[str, float, float]] = []
    for ln in lines[header_idx + 1 :]:
        p = ln.split(";")
        if len(p) < len(cols):
            continue
        ticker = p[ix["TckrSymb"]]
        if p[ix["SgmtNm"]] != "FINANCIAL" or not _TICKER_RE.match(ticker):
            continue
        pu = _to_float(p[ix["AdjstdQt"]])
        taxa = _to_float(p[ix["AdjstdQtTax"]])
        if pu is None or taxa is None:
            continue
        if refdate is None:
            refdate = date.fromisoformat(p[ix["RptDt"]])
        rows.append((ticker, pu, taxa))
    return refdate, rows


def fetch(data: str | date):
    import requests

    d = data.isoformat() if isinstance(data, date) else data
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    meta = s.get(
        f"{_B3_API}/requestname",
        params={"fileName": "TradeInformationConsolidatedFile", "date": d},
        timeout=60,
    )
    if meta.status_code != 200 or "token" not in meta.text:
        raise NoSettlementError(d)
    token = meta.json()["token"]
    r = s.get(f"{_B3_API}/", params={"token": token}, timeout=120)
    if r.status_code != 200 or len(r.content) < 1000:
        raise NoSettlementError(d)
    return r.content.decode("latin-1")


def _previous_business_day(d: date) -> date:
    prev = d - timedelta(days=1)
    while not is_business_day(prev):
        prev -= timedelta(days=1)
    return prev


def fetch_recent(start: date) -> tuple[date, list[tuple[str, float, float]]]:
    d = start
    for _ in range(MAX_LOOKBACK):
        try:
            refdate, rows = parse_trade_information(fetch(d))
            if refdate is not None and rows:
                return refdate, rows
            raise NoSettlementError(d.isoformat())
        except NoSettlementError:
            print(f"[fetch_di1] no settlements on {d.isoformat()}, going back…")
            d = _previous_business_day(d)
    raise SystemExit(f"no settlements in the last {MAX_LOOKBACK} business days from {start}")


def save(refdate: date, rows, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"settlements_{refdate.isoformat()}.csv"
    enriched = []
    for ticker, pu, taxa in rows:
        du = DI1(ticker, refdate).du
        spot = rate_from_pu(pu, du)
        enriched.append((ticker, du, pu, taxa, spot))
    enriched.sort(key=lambda r: r[1])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["refdate", "ticker", "business_days", "settlement_pu", "settlement_rate", "spot"]
        )
        for ticker, du, pu, taxa, spot in enriched:
            w.writerow(
                [refdate.isoformat(), ticker, du, f"{pu:.5f}", f"{taxa / 100:.10f}", f"{spot:.10f}"]
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DI1 settlement prices (per-contract adjusted price)."
    )
    parser.add_argument("data", nargs="?", help="date YYYY-MM-DD (default: recent)")
    args = parser.parse_args()

    start = date.fromisoformat(args.data) if args.data else preceding(date.today())
    refdate, rows = fetch_recent(start)
    path = save(refdate, rows)
    dus = [DI1(t, refdate).du for t, _, _ in rows]
    print(
        f"[fetch_di1] d0={refdate.isoformat()}  DI1 contracts={len(rows)}  "
        f"du={min(dus)}..{max(dus)}"
    )
    print(f"[fetch_di1] saved to {path}")


if __name__ == "__main__":
    main()
