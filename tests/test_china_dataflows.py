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
def test_china_stock_data_uses_local_vendor(monkeypatch):
    monkeypatch.setattr(china, "_fetch_akshare_ohlcv", lambda *a, **k: _sample_ohlcv("AKShare unit"))

    out = china.get_china_stock_data("600519.SS", "2025-06-03", "2025-06-05")

    assert "China A-share stock data for 600519.SH" in out
    assert "Source: AKShare unit" in out
    assert "2025-06-05" in out
    assert "11.2" in out


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
