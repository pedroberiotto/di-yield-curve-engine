import fetch_anbima_ettj as fa
import pytest


class _FakeDF:

    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        yield from enumerate(self._rows)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("14,0405", 14.0405),
        ("0,141307346513404", 0.141307346513404),
        ("1,56583639902359E-03", 1.56583639902359e-3),
        ("-1,78318333731498E-02", -1.78318333731498e-2),
    ],
)
def test_to_float(raw, expected):
    assert fa._to_float(raw) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "raw, expected",
    [("126", 126), ("1.008", 1008), ("12.477", 12477), (" 252 ", 252)],
)
def test_to_du_handles_thousands_separator(raw, expected):
    assert fa._to_du(raw) == expected


@pytest.mark.parametrize("blank", ["", "   ", "nan", "None", "-"])
def test_is_blank_true(blank):
    assert fa._is_blank(blank)


@pytest.mark.parametrize("filled", ["14,04", "0,0", "-1,2"])
def test_is_blank_false(filled):
    assert not fa._is_blank(filled)


def test_parse_pre_vertices_skips_blanks_and_sorts_and_scales():
    df = _FakeDF(
        [
            {"Vertice": "252", "Prefixados": "14,4302"},
            {"Vertice": "126", "Prefixados": "14,0405"},
            {"Vertice": "1.008", "Prefixados": ""},
            {"Vertice": "504", "Prefixados": "14,8655"},
        ]
    )
    du, rate = fa.parse_pre_vertices(df)
    assert du == [126, 252, 504]
    assert rate[0] == pytest.approx(0.140405)
    assert rate[1] == pytest.approx(0.144302)
    assert all(0 < r < 1 for r in rate)


def test_br_to_iso():
    from datetime import date

    assert fa._br_to_iso("19/06/2026") == date(2026, 6, 19)


_TRADE_INFO = """Status do Arquivo: Final
RptDt;TckrSymb;ISIN;SgmtNm;MinPric;MaxPric;TradAvrgPric;LastPric;OscnPctg;AdjstdQt;AdjstdQtTax;RefPric;TradQty;FinInstrmQty;NtlFinVol
2026-06-18;DI1N26;BR;FINANCIAL;;;;;;99528,37;14,153;;;;
2026-06-18;DI1F27;BR;FINANCIAL;;;;;;93064,98;14,245;;;;
2026-06-18;PETR4;BR;CASH;;;;;;;;;;;
2026-06-18;DI1Z99;BR;FINANCIAL;;;;;;;;;;;
"""


def test_parse_trade_information_filters_di1_with_adjustment():
    import fetch_di1

    refdate, rows = fetch_di1.parse_trade_information(_TRADE_INFO)
    from datetime import date

    assert refdate == date(2026, 6, 18)
    assert [r[0] for r in rows] == ["DI1N26", "DI1F27"]
    assert rows[0] == ("DI1N26", 99528.37, 14.153)


def test_parse_trade_information_requires_header():
    import fetch_di1

    with pytest.raises(ValueError, match="header"):
        fetch_di1.parse_trade_information("lixo\nsem header\n")


@pytest.mark.parametrize(
    "raw, expected",
    [("99528,37", 99528.37), ("14,245", 14.245), ("1.234,56", 1234.56), ("", None), ("-", None)],
)
def test_fetch_di1_to_float(raw, expected):
    import fetch_di1

    assert fetch_di1._to_float(raw) == expected


def test_parse_bcb_overnight_decimal():
    import fetch_di1

    payload = [{"data": "18/06/2026", "valor": "14.65"}]
    assert fetch_di1.parse_bcb_overnight(payload) == pytest.approx(0.1465)


def test_parse_bcb_overnight_comma_and_last_wins():
    import fetch_di1

    payload = [{"data": "17/06/2026", "valor": "14,60"}, {"data": "18/06/2026", "valor": "14,65"}]
    assert fetch_di1.parse_bcb_overnight(payload) == pytest.approx(0.1465)


def test_parse_bcb_overnight_empty():
    import fetch_di1

    assert fetch_di1.parse_bcb_overnight([]) is None


def test_live_contracts_drops_contract_maturing_on_d0():
    from datetime import date

    import fetch_di1

    d0 = date(2026, 7, 1)
    rows = [
        ("DI1N26", 100000.0, 14.0),
        ("DI1Q26", 99000.0, 14.1),
        ("DI1F27", 93000.0, 14.2),
    ]
    live = fetch_di1.live_contracts(d0, rows)
    assert [r[0] for r in live] == ["DI1Q26", "DI1F27"]


def test_save_skips_expired_front(tmp_path):
    from datetime import date

    import fetch_di1

    d0 = date(2026, 7, 1)
    rows = [
        ("DI1N26", 100000.0, 14.0),
        ("DI1Q26", 99000.0, 14.1),
    ]
    path = fetch_di1.save(d0, rows, tmp_path)
    tickers = [ln.split(",")[1] for ln in path.read_text().splitlines()[1:]]
    assert tickers == ["DI1Q26"]
