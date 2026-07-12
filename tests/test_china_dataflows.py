import copy
from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import china, interface, market_data_validator
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _sample_ohlcv(source="unit"):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-06-03", "2025-06-04", "2025-06-05"]),
            "Open": [10.0, 10.5, 11.0],
            "High": [10.8, 11.1, 11.5],
            "Low": [9.8, 10.3, 10.9],
            "Close": [10.6, 10.9, 11.2],
            "Volume": [1000, 1200, 1300],
            "PE_TTM": [12.0, 12.2, 12.4],
            "PB_MRQ": [1.5, 1.6, 1.7],
        }
    )
    df.attrs["source"] = source
    return df


@pytest.mark.unit
def test_parse_a_share_symbol_common_forms():
    sh = china.parse_a_share_symbol("600519.SS")
    assert sh.code == "600519"
    assert sh.market == "sh"
    assert sh.tushare == "600519.SH"
    assert sh.baostock == "sh.600519"

    sz = china.parse_a_share_symbol("SZ000333")
    assert sz.code == "000333"
    assert sz.market == "sz"
    assert sz.yahoo == "000333.SZ"

    assert china.parse_a_share_symbol("AAPL") is None


@pytest.mark.unit
def test_china_index_symbols_are_not_treated_as_stocks():
    index = china.parse_china_index_symbol("000001.SS")

    assert index.name == "SSE Composite Index"
    assert index.akshare == "sh000001"
    assert china.is_china_index_symbol("000001.SS")
    assert not china.is_a_share_symbol("000001.SS")
    assert china.is_a_share_symbol("000001.SZ")


@pytest.mark.unit
def test_china_benchmark_resolution_is_board_aware():
    assert china.resolve_china_benchmark("600519.SS") == "000001.SS"
    assert china.resolve_china_benchmark("688981.SS") == "000001.SS"
    assert china.resolve_china_benchmark("000333.SZ") == "399001.SZ"
    assert china.resolve_china_benchmark("300750.SZ") == "399006.SZ"
    assert china.resolve_china_benchmark("AAPL") is None


@pytest.mark.unit
def test_china_ohlcv_vendor_chain_prefers_tushare_only_when_configured(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_KEY", raising=False)
    assert china.china_ohlcv_vendor_chain() == ("AKShare", "BaoStock")

    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    assert china.china_ohlcv_vendor_chain() == ("Tushare", "AKShare", "BaoStock")


@pytest.mark.unit
def test_china_stock_data_uses_local_vendor(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_KEY", raising=False)
    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", lambda *a, **k: _sample_ohlcv("AKShare unit"))

    out = china.get_china_stock_data("600519.SS", "2025-06-03", "2025-06-05")

    assert "China A-share stock data for 600519.SH" in out
    assert "Source: AKShare unit" in out
    assert "2025-06-05" in out
    assert "11.2" in out


@pytest.mark.unit
def test_china_ohlcv_cache_reuses_successful_fetch(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_KEY", raising=False)
    calls = []

    def fetch_akshare(*args, **kwargs):
        calls.append(args)
        return _sample_ohlcv("AKShare unit")

    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", fetch_akshare)
    first = china.load_china_ohlcv_range("600519.SS", "2025-06-03", "2025-06-05")

    assert first.attrs["source"] == "AKShare unit"
    assert len(calls) == 1

    def fail_live_fetch(*args, **kwargs):
        raise AssertionError("complete cache hit should not call live vendors")

    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", fail_live_fetch)
    monkeypatch.setattr(china, "_fetch_baostock_ohlcv", fail_live_fetch)

    second = china.load_china_ohlcv_range("600519.SS", "2025-06-03", "2025-06-05")

    assert list(second["Close"]) == [10.6, 10.9, 11.2]
    assert second.attrs["source"] == "China OHLCV cache (600519.SH)"


@pytest.mark.unit
def test_china_ohlcv_cache_falls_back_to_partial_rows(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_KEY", raising=False)
    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", lambda *a, **k: _sample_ohlcv("AKShare unit"))

    china.load_china_ohlcv_range("600519.SS", "2025-06-03", "2025-06-05")

    def unavailable(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", unavailable)
    monkeypatch.setattr(china, "_fetch_baostock_ohlcv", unavailable)

    fallback = china.load_china_ohlcv_range("600519.SS", "2025-06-04", "2025-06-10")

    assert list(fallback["Date"].dt.strftime("%Y-%m-%d")) == ["2025-06-04", "2025-06-05"]
    assert "partial fallback" in fallback.attrs["source"]


@pytest.mark.unit
def test_china_fundamentals_text_cache_reuses_successful_report(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    calls = {"basic": 0, "financial": 0}

    def fake_basic(symbol):
        calls["basic"] += 1
        return {"Name": "Unit Co", "IPO Date": "2000-01-01"}

    def fake_financial(symbol, kind, curr_date, freq="quarterly", limit=8):
        calls["financial"] += 1
        return pd.DataFrame(
            [
                {
                    "pubDate": pd.Timestamp("2025-04-30"),
                    "statDate": pd.Timestamp("2025-03-31"),
                    "roeAvg": 12.3,
                    "currentRatio": 1.5,
                    "CFOToNP": 0.9,
                }
            ]
        )

    monkeypatch.setattr(china, "_fetch_baostock_basic", fake_basic)
    monkeypatch.setattr(china, "_fetch_akshare_static_info", lambda symbol: {})
    monkeypatch.setattr(china, "_latest_baostock_valuation", lambda symbol, curr_date: {"PE TTM": "10"})
    monkeypatch.setattr(china, "_fetch_tushare_daily_basic", lambda symbol, curr_date: {})
    monkeypatch.setattr(china, "_fetch_baostock_financial_rows", fake_financial)

    first = china.get_china_fundamentals("600519.SS", "2025-06-05")
    second = china.get_china_fundamentals("600519.SS", "2025-06-05")

    assert first == second
    assert "Unit Co" in second
    assert calls == {"basic": 1, "financial": 3}


@pytest.mark.unit
def test_china_statement_text_cache_reuses_successful_report(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    calls = {"statement": 0}

    def fake_statement(symbol, statement_symbol, curr_date, freq="quarterly", limit=8):
        calls["statement"] += 1
        return pd.DataFrame([{"announcement": "2025-04-30", "value": 1.0}])

    monkeypatch.setattr(china, "_fetch_akshare_statement", fake_statement)

    first = china.get_china_balance_sheet("600519.SS", "quarterly", "2025-06-05")
    second = china.get_china_balance_sheet("600519.SS", "quarterly", "2025-06-05")

    assert first == second
    assert "Balance Sheet" in second
    assert calls["statement"] == 1


@pytest.mark.unit
def test_china_stock_data_rejects_non_a_share():
    with pytest.raises(NoMarketDataError):
        china.get_china_stock_data("AAPL", "2025-06-03", "2025-06-05")


@pytest.mark.unit
def test_interface_registers_china_vendor():
    for method in (
        "get_stock_data",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_news",
    ):
        assert "china" in interface.VENDOR_METHODS[method]


@pytest.mark.unit
def test_default_vendor_chain_prefers_china_for_stock_data():
    _reset_config()
    calls = []

    def china_vendor(symbol, *args, **kwargs):
        calls.append(symbol)
        return "CHINA_DATA"

    with mock.patch.dict(
        interface.VENDOR_METHODS,
        {"get_stock_data": {"china": china_vendor, "yfinance": lambda *a, **k: "YF_DATA"}},
        clear=False,
    ):
        result = interface.route_to_vendor("get_stock_data", "600519.SS", "2025-06-03", "2025-06-05")

    assert result == "CHINA_DATA"
    assert calls == ["600519.SS"]
    _reset_config()


@pytest.mark.unit
def test_market_snapshot_uses_china_loader_for_a_share(monkeypatch):
    monkeypatch.setattr(market_data_validator, "load_china_ohlcv", lambda *a, **k: _sample_ohlcv())

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("A-share snapshot should not call yfinance loader")

    monkeypatch.setattr(market_data_validator, "load_ohlcv", fail_yfinance)

    snap = market_data_validator.build_verified_market_snapshot("600519.SS", "2025-06-05")

    assert "Verified market data snapshot for 600519.SS" in snap
    assert "| Close | 11.20 |" in snap


@pytest.mark.unit
def test_partial_config_defaults_keep_china_chain():
    _reset_config()
    set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})

    fresh = config_module.get_config()

    assert fresh["data_vendors"]["core_stock_apis"] == "alpha_vantage"
    assert fresh["data_vendors"]["technical_indicators"] == "china,yfinance"
    assert fresh["data_vendors"]["fundamental_data"] == "china,yfinance"
    assert fresh["data_vendors"]["news_data"] == "china,yfinance"
    _reset_config()
