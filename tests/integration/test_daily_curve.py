from datetime import date

import daily_curve as dc
import numpy as np
import pytest

from yieldcurve.bootstrap import Fixture


def _fixture(n=60, mutate=None) -> Fixture:
    du = np.arange(1, n + 1)
    rate = np.full(n, 0.14)
    if mutate:
        mutate(rate)
    return Fixture(date(2026, 6, 19), du, rate)


def test_recent_business_days():
    days = dc.recent_business_days(date(2026, 6, 19), 3)
    assert days == [date(2026, 6, 19), date(2026, 6, 18), date(2026, 6, 17)]


def test_recent_business_days_skips_weekend():
    assert dc.recent_business_days(date(2026, 6, 21), 1) == [date(2026, 6, 19)]


def test_curve_exists(tmp_path):
    (tmp_path / "curve_2026-06-19.csv").write_text("x")
    assert dc.curve_exists(date(2026, 6, 19), tmp_path)
    assert not dc.curve_exists(date(2026, 6, 18), tmp_path)


def test_pending_days_curve_without_benchmark_is_still_pending(tmp_path):
    (tmp_path / "curve_2026-06-19.csv").write_text("x")
    pend = dc.pending_days(date(2026, 6, 19), n=3, curves_dir=tmp_path)
    assert pend == [date(2026, 6, 17), date(2026, 6, 18), date(2026, 6, 19)]


def test_pending_days_complete_day_is_skipped(tmp_path):
    (tmp_path / "curve_2026-06-19.csv").write_text("x")
    (tmp_path / "benchmark_2026-06-19.csv").write_text("x")
    pend = dc.pending_days(date(2026, 6, 19), n=3, curves_dir=tmp_path)
    assert pend == [date(2026, 6, 17), date(2026, 6, 18)]


def test_pending_days_without_benchmark_only_needs_curve(tmp_path):
    (tmp_path / "curve_2026-06-19.csv").write_text("x")
    pend = dc.pending_days(date(2026, 6, 19), n=3, curves_dir=tmp_path, with_benchmark=False)
    assert pend == [date(2026, 6, 17), date(2026, 6, 18)]


def test_check_refdate_ok():
    dc.check_refdate(date(2026, 6, 19), date(2026, 6, 19))


def test_check_refdate_mismatch():
    with pytest.raises(ValueError):
        dc.check_refdate(date(2026, 6, 18), date(2026, 6, 19))


def test_sanity_check_passes_on_real_fixture(settlements_fixture):
    dc.sanity_check(settlements_fixture, min_vertices=20)


def test_sanity_check_too_few_vertices(settlements_fixture):
    with pytest.raises(ValueError, match="too few vertices"):
        dc.sanity_check(settlements_fixture, min_vertices=10_000)


def test_sanity_check_no_overnight():
    fx = Fixture(date(2026, 6, 19), np.arange(2, 62), np.full(60, 0.14))
    with pytest.raises(ValueError, match="overnight"):
        dc.sanity_check(fx, min_vertices=10)


def test_sanity_check_nan():
    fx = _fixture(mutate=lambda r: r.__setitem__(5, np.nan))
    with pytest.raises(ValueError, match="NaN"):
        dc.sanity_check(fx, min_vertices=10)


@pytest.mark.parametrize("bad", [1.5, 0.0, -0.1])
def test_sanity_check_rate_out_of_range(bad):
    fx = _fixture(mutate=lambda r: r.__setitem__(5, bad))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        dc.sanity_check(fx, min_vertices=10)


_CONTRACTS = [
    ("DI1N26", "99528,37", "14,153"),
    ("DI1F27", "93064,98", "14,245"),
    ("DI1F28", "80970,00", "14,735"),
]
_HEADER = (
    "Status do Arquivo: Final\n"
    "RptDt;TckrSymb;ISIN;SgmtNm;MinPric;MaxPric;TradAvrgPric;LastPric;OscnPctg;"
    "AdjstdQt;AdjstdQtTax;RefPric;TradQty;FinInstrmQty;NtlFinVol\n"
)


def _trade_info(refdate="2026-06-18") -> str:
    body = "".join(
        f"{refdate};{tk};BR;FINANCIAL;;;;;;{pu};{tx};;;;\n" for tk, pu, tx in _CONTRACTS
    )
    return _HEADER + body


def _stub(monkeypatch, refdate="2026-06-18"):
    monkeypatch.setattr(dc.fetch_di1, "fetch", lambda target: _trade_info(refdate))
    monkeypatch.setattr(dc.fetch_di1, "fetch_overnight", lambda d: None)


def test_process_day_ok(tmp_path, monkeypatch):
    _stub(monkeypatch)
    status = dc.process_day(
        date(2026, 6, 18),
        curves_dir=tmp_path, raw_dir=tmp_path, min_vertices=2, with_benchmark=False,
    )
    assert status == "ok"
    assert (tmp_path / "curve_2026-06-18.csv").exists()
    assert (tmp_path / "settlements_2026-06-18.csv").exists()
    assert (tmp_path / "risk_2026-06-18.csv").exists()


def test_process_day_refdate_mismatch_writes_nothing(tmp_path, monkeypatch):
    _stub(monkeypatch, refdate="2026-06-18")
    status = dc.process_day(
        date(2026, 6, 17),
        curves_dir=tmp_path, raw_dir=tmp_path, min_vertices=2, with_benchmark=False,
    )
    assert status.startswith("not-published")
    assert not (tmp_path / "curve_2026-06-17.csv").exists()


def test_process_day_no_data(tmp_path, monkeypatch):
    def boom(target):
        raise RuntimeError("sem dados publicados")

    monkeypatch.setattr(dc.fetch_di1, "fetch", boom)
    status = dc.process_day(
        date(2026, 6, 18), curves_dir=tmp_path, raw_dir=tmp_path, with_benchmark=False
    )
    assert status.startswith("no-data")
    assert not any(tmp_path.glob("curve_*.csv"))


def test_process_day_sanity_failure(tmp_path, monkeypatch):
    _stub(monkeypatch)
    status = dc.process_day(
        date(2026, 6, 18),
        curves_dir=tmp_path,
        raw_dir=tmp_path,
        min_vertices=10_000,
        with_benchmark=False,
    )
    assert status.startswith("sanity-failed")
    assert not (tmp_path / "curve_2026-06-18.csv").exists()


def test_run_builds_then_idempotent(tmp_path, monkeypatch):
    _stub(monkeypatch)
    res = dc.run(
        date(2026, 6, 18), n=1, curves_dir=tmp_path, raw_dir=tmp_path,
        min_vertices=2, with_benchmark=False,
    )
    assert res == [(date(2026, 6, 18), "ok")]
    assert (tmp_path / "curve_2026-06-18.csv").exists()

    res2 = dc.run(
        date(2026, 6, 18), n=1, curves_dir=tmp_path, raw_dir=tmp_path,
        min_vertices=2, with_benchmark=False,
    )
    assert res2 == []


def _write_settlements(path, refdate="2026-06-18"):
    path.write_text(
        "refdate,ticker,business_days,settlement_pu,settlement_rate,spot\n"
        f"{refdate},DI1N26,9,99528.37,0.14153,0.14153\n"
        f"{refdate},DI1F27,136,93064.98,0.14245,0.14245\n"
        f"{refdate},DI1F28,387,80970.00,0.14735,0.14735\n"
    )


def test_process_day_completes_benchmark_without_refetch(tmp_path, monkeypatch):
    curves, raw = tmp_path / "c", tmp_path / "r"
    curves.mkdir()
    raw.mkdir()
    _write_settlements(raw / "settlements_2026-06-18.csv")
    (curves / "curve_2026-06-18.csv").write_text(
        "refdate,business_days,zero,df\n2026-06-18,1,0.14,0.99\n"
    )

    def _no_b3(target):
        raise AssertionError("must not hit B3 when the curve already exists")

    monkeypatch.setattr(dc.fetch_di1, "fetch", _no_b3)
    import fetch_anbima_ettj

    monkeypatch.setattr(fetch_anbima_ettj, "fetch", lambda d: (date(2026, 6, 18), [126], [0.14]))

    status = dc.process_day(date(2026, 6, 18), curves_dir=curves, raw_dir=raw)
    assert status == "benchmark-ok"
    assert (curves / "benchmark_2026-06-18.csv").exists()


def test_process_day_skips_when_complete(tmp_path, monkeypatch):
    curves, raw = tmp_path / "c", tmp_path / "r"
    curves.mkdir()
    raw.mkdir()
    _write_settlements(raw / "settlements_2026-06-18.csv")
    (curves / "curve_2026-06-18.csv").write_text("x")
    (curves / "benchmark_2026-06-18.csv").write_text("x")

    def _no_b3(target):
        raise AssertionError("complete day must not trigger fetch")

    monkeypatch.setattr(dc.fetch_di1, "fetch", _no_b3)
    status = dc.process_day(date(2026, 6, 18), curves_dir=curves, raw_dir=raw)
    assert status == "already-complete"
