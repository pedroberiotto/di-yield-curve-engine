import csv

import build_curve
import pytest


def test_build_from_fixture_passes_through_nodes(settlements_path):
    curve = build_curve.build(settlements_path)
    du, zero = curve.node_rates()
    for d, z in zip(du, zero, strict=True):
        assert curve.zero(int(d)) == pytest.approx(z, abs=1e-10)


def test_save_curve_writes_nodes_and_latest(settlements_path, tmp_path):
    curve = build_curve.build(settlements_path)
    path = build_curve.save_curve(curve, out_dir=tmp_path)
    assert path.exists()
    assert (tmp_path / "latest.csv").exists()
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert [c for c in rows[0]] == ["refdate", "business_days", "zero", "df"]
    assert len(rows) == curve.node_du.size - 1
    du0 = int(rows[0]["business_days"])
    assert float(rows[0]["df"]) == pytest.approx(curve.df(du0), abs=1e-10)


def test_latest_points_to_newest_even_after_backfill(settlements_path, tmp_path):
    curve = build_curve.build(settlements_path)
    build_curve.save_curve(curve, out_dir=tmp_path)

    older = tmp_path / "curve_2026-06-17.csv"
    older.write_text("refdate,business_days,zero,df\n2026-06-17,1,0.14,0.99\n")
    build_curve.update_latest(tmp_path)

    latest = (tmp_path / "latest.csv").read_text()
    newest = (tmp_path / f"curve_{curve.d0.isoformat()}.csv").read_text()
    assert latest == newest


def test_benchmark_filters_out_of_range_and_computes_bps(settlements_path):
    curve = build_curve.build(settlements_path)
    du0 = int(curve.node_du[1])
    du_max = int(curve.node_du[-1])
    di = curve.zero(du0)
    anbima = di - 10e-4

    rows = build_curve.benchmark(
        curve,
        du_list=[du0, du_max + 100],
        rate_list=[anbima, 0.10],
    )
    assert len(rows) == 1
    du, di_r, an_r, bps = rows[0]
    assert du == du0
    assert di_r == pytest.approx(di, abs=1e-12)
    assert an_r == pytest.approx(anbima, abs=1e-12)
    assert bps == pytest.approx(10.0, abs=1e-6)


def test_benchmark_summary_stats():
    rows = [(126, 0.142, 0.140, 20.0), (252, 0.146, 0.147, -10.0)]
    s = build_curve.benchmark_summary(rows)
    assert s["n"] == 2
    assert s["mean_abs_bps"] == pytest.approx(15.0)
    assert s["max_abs_bps"] == pytest.approx(20.0)


def test_benchmark_summary_empty():
    s = build_curve.benchmark_summary([])
    assert s["n"] == 0


def test_save_benchmark_writes_report(tmp_path):
    from datetime import date

    rows = [(126, 0.142, 0.140, 20.0), (252, 0.146, 0.147, -10.0)]
    path = build_curve.save_benchmark(rows, date(2026, 6, 19), out_dir=tmp_path)
    assert path.exists()
    with open(path) as fh:
        out = list(csv.DictReader(fh))
    assert [c for c in out[0]] == ["business_days", "rate_di", "rate_ref", "diff_bps"]
    assert len(out) == 2


def test_reference_risk_ladder(settlements_path):
    from yieldcurve.risk import dv01

    curve = build_curve.build(settlements_path)
    kr = build_curve.reference_risk(curve)
    du, _ = curve.node_rates()
    assert list(kr.du) == list(du)
    assert all(v > 0 for v in kr.dv01)
    book = [(int(d), build_curve.NOTIONAL) for d in du]
    assert kr.total == pytest.approx(dv01(curve, book), rel=1e-6)


def test_save_risk_writes_ladder(settlements_path, tmp_path):
    curve = build_curve.build(settlements_path)
    path = build_curve.save_risk(build_curve.reference_risk(curve), curve.d0, out_dir=tmp_path)
    assert path.exists()
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert [c for c in rows[0]] == ["refdate", "business_days", "dv01"]
    assert len(rows) == curve.node_du.size - 1
