from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from yieldcurve.bootstrap import build_curve, load_settlements
from yieldcurve.curve import Curve
from yieldcurve.risk import dv01, key_rate_dv01, pv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

app = FastAPI(title="DI Yield Curve API", version="0.1.0")


def _latest_settlements() -> Path:
    files = sorted(RAW_DIR.glob("settlements_*.csv"))
    if not files:
        raise FileNotFoundError(
            "no settlements settlements_*.csv in data/raw (run scripts/fetch_di1.py)"
        )
    return files[-1]


@lru_cache
def _build_default_curve() -> Curve:
    return build_curve(load_settlements(_latest_settlements()))


def provide_curve() -> Curve:
    try:
        return _build_default_curve()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _resolve_term(du: int | None, date: str | None):
    if du is not None:
        return du
    if date is not None:
        return date
    raise HTTPException(status_code=422, detail="provide 'du' or 'date'")


class CurveMeta(BaseModel):
    d0: str
    calendar: str
    interpolation: str
    n_nodes: int
    du_min: int
    du_max: int


class Node(BaseModel):
    du: int
    zero: float
    df: float


class TermValue(BaseModel):
    du: int
    value: float


class ForwardValue(BaseModel):
    du1: int
    du2: int
    forward: float


class CashFlowIn(BaseModel):
    du: int = Field(gt=0)
    amount: float


class PortfolioIn(BaseModel):
    cashflows: list[CashFlowIn] = Field(min_length=1)
    bump: float = 1e-4


class KeyRateItem(BaseModel):
    du: int
    dv01: float


class RiskOut(BaseModel):
    pv: float
    dv01: float
    total_key_rate: float
    key_rate: list[KeyRateItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/curve", response_model=CurveMeta)
def curve_meta(curve: Curve = Depends(provide_curve)) -> CurveMeta:
    du, _ = curve.node_rates()
    return CurveMeta(
        d0=curve.d0.isoformat(),
        calendar=curve.calendar,
        interpolation=curve.interpolation,
        n_nodes=int(du.size),
        du_min=int(du[0]),
        du_max=int(du[-1]),
    )


@app.get("/curve/nodes", response_model=list[Node])
def curve_nodes(curve: Curve = Depends(provide_curve)) -> list[Node]:
    du, zero = curve.node_rates()
    return [
        Node(du=int(d), zero=float(z), df=float(curve.df(int(d))))
        for d, z in zip(du, zero, strict=True)
    ]


@app.get("/curve/df", response_model=TermValue)
def curve_df(
    du: int | None = None,
    date: str | None = None,
    curve: Curve = Depends(provide_curve),
) -> TermValue:
    term = _resolve_term(du, date)
    try:
        resolved = int(curve.du(term))
        return TermValue(du=resolved, value=curve.df(term))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/curve/zero", response_model=TermValue)
def curve_zero(
    du: int | None = None,
    date: str | None = None,
    curve: Curve = Depends(provide_curve),
) -> TermValue:
    term = _resolve_term(du, date)
    try:
        resolved = int(curve.du(term))
        return TermValue(du=resolved, value=curve.zero(term))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/curve/forward", response_model=ForwardValue)
def curve_forward(
    du1: int | None = None,
    du2: int | None = None,
    date1: str | None = None,
    date2: str | None = None,
    curve: Curve = Depends(provide_curve),
) -> ForwardValue:
    t1 = _resolve_term(du1, date1)
    t2 = _resolve_term(du2, date2)
    try:
        return ForwardValue(
            du1=int(curve.du(t1)),
            du2=int(curve.du(t2)),
            forward=curve.forward(t1, t2),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/risk/dv01", response_model=RiskOut)
def risk_dv01(
    body: PortfolioIn,
    curve: Curve = Depends(provide_curve),
) -> RiskOut:
    cashflows = [(cf.du, cf.amount) for cf in body.cashflows]
    kr = key_rate_dv01(curve, cashflows, bump=body.bump)
    return RiskOut(
        pv=pv(curve, cashflows),
        dv01=dv01(curve, cashflows, bump=body.bump),
        total_key_rate=kr.total,
        key_rate=[
            KeyRateItem(du=int(d), dv01=float(v))
            for d, v in zip(kr.du, kr.dv01, strict=True)
        ],
    )
