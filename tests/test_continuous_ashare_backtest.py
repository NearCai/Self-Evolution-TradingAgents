import pandas as pd
import pytest

from scripts import run_continuous_backtest_ashare as continuous


def _sample_history(source: str = "unit"):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"]),
            "Open": [10.0, 10.2, 10.4],
            "High": [10.5, 10.7, 10.8],
            "Low": [9.8, 10.0, 10.2],
            "Close": [10.1, 10.3, 10.6],
            "Volume": [1000, 1100, 1200],
        }
    )
    df.attrs["source"] = source
    return df


@pytest.mark.unit
def test_history_with_retry_uses_china_only_for_a_share(monkeypatch):
    monkeypatch.setattr(
        continuous,
        "load_china_ohlcv_range",
        lambda *args, **kwargs: _sample_history("AKShare unit"),
    )

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("A-share history must not fall back to yfinance")

    monkeypatch.setattr(continuous.yf, "Ticker", fail_yfinance)

    history = continuous.history_with_retry("600519.SS", "2026-06-01", "2026-06-04")

    assert list(history.index.strftime("%Y-%m-%d")) == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert continuous.history_source(history) == "AKShare unit"


@pytest.mark.unit
def test_history_with_retry_does_not_fallback_to_yfinance_when_china_fails(monkeypatch):
    monkeypatch.setattr(
        continuous,
        "load_china_ohlcv_range",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("china unavailable")),
    )

    def fail_yfinance(*args, **kwargs):
        raise AssertionError("A-share failure must stay on the China data path")

    monkeypatch.setattr(continuous.yf, "Ticker", fail_yfinance)

    with pytest.raises(RuntimeError, match="china unavailable"):
        continuous.history_with_retry("000333.SZ", "2026-06-01", "2026-06-04", retries=0)


@pytest.mark.unit
def test_resolve_benchmark_is_board_aware_for_chinext():
    assert continuous.resolve_benchmark("600519.SS", {}) == "000001.SS"
    assert continuous.resolve_benchmark("000333.SZ", {}) == "399001.SZ"
    assert continuous.resolve_benchmark("300750.SZ", {}) == "399006.SZ"


@pytest.mark.unit
def test_build_backtest_config_forces_china_data_vendors(tmp_path):
    cfg = continuous.build_backtest_config(
        output_dir=tmp_path,
        memory_mode="experiment",
        memory_holding_days=5,
    )

    assert cfg["data_vendors"]["core_stock_apis"] == "china"
    assert cfg["data_vendors"]["technical_indicators"] == "china"
    assert cfg["data_vendors"]["fundamental_data"] == "china"
    assert cfg["data_vendors"]["news_data"] == "china"
    assert cfg["benchmark_ticker"] is None
    assert cfg["memory_lookahead_safe"] is True


@pytest.mark.unit
def test_build_backtest_config_can_disable_optional_overseas_sources(tmp_path):
    cfg = continuous.build_backtest_config(
        output_dir=tmp_path,
        memory_mode="experiment",
        memory_holding_days=5,
        enable_prediction_markets=False,
        enable_us_social_sources=False,
    )

    assert cfg["enable_prediction_markets"] is False
    assert cfg["enable_us_social_sources"] is False


@pytest.mark.unit
def test_build_backtest_config_can_enable_evolution_skills(tmp_path):
    skills_path = tmp_path / "candidate_skills.jsonl"
    cfg = continuous.build_backtest_config(
        output_dir=tmp_path,
        memory_mode="experiment",
        memory_holding_days=5,
        evolution_skills_path=str(skills_path),
        evolution_skill_max_skills=2,
        evolution_skill_max_chars=900,
        evolution_skill_allowed_types=["opportunity", "promote"],
        evolution_opportunity_gate_enabled=True,
    )

    assert cfg["evolution_skills_path"] == str(skills_path)
    assert cfg["evolution_skill_max_skills"] == 2
    assert cfg["evolution_skill_max_chars"] == 900
    assert cfg["evolution_skill_allowed_types"] == ["opportunity", "promote"]
    assert cfg["evolution_opportunity_gate_enabled"] is True


@pytest.mark.unit
def test_build_opportunity_evidence_requires_positive_signals():
    stock_prices = {
        "2026-06-01": 10.0,
        "2026-06-02": 9.8,
        "2026-06-03": 9.6,
        "2026-06-04": 9.5,
        "2026-06-05": 9.4,
    }
    benchmark_prices = {
        "2026-06-01": 100.0,
        "2026-06-02": 101.0,
        "2026-06-03": 102.0,
        "2026-06-04": 103.0,
        "2026-06-05": 104.0,
    }

    evidence = continuous.build_opportunity_evidence(
        stock_prices,
        benchmark_prices,
        "2026-06-05",
        enabled=True,
        lookback_days=4,
        min_positive_signals=2,
    )

    assert evidence["allow_opportunity"] is False
    assert evidence["positive_signal_count"] < 2
    assert evidence["reason"] == "insufficient_positive_evidence"


@pytest.mark.unit
def test_build_opportunity_evidence_allows_constructive_setup():
    stock_prices = {
        "2026-06-01": 10.0,
        "2026-06-02": 10.2,
        "2026-06-03": 10.4,
        "2026-06-04": 10.5,
        "2026-06-05": 10.7,
    }
    benchmark_prices = {
        "2026-06-01": 100.0,
        "2026-06-02": 100.1,
        "2026-06-03": 100.2,
        "2026-06-04": 100.1,
        "2026-06-05": 100.0,
    }

    evidence = continuous.build_opportunity_evidence(
        stock_prices,
        benchmark_prices,
        "2026-06-05",
        enabled=True,
        lookback_days=4,
        min_positive_signals=2,
    )

    assert evidence["allow_opportunity"] is True
    assert evidence["positive_signal_count"] >= 2
    assert evidence["stock_return_lookback"] > 0
    assert evidence["relative_return_lookback"] > 0


@pytest.mark.unit
def test_default_output_dir_is_project_local():
    out = continuous.default_output_dir("2026-06-01")

    assert out == continuous.PROJECT_ROOT / "results" / "continuous_ashare_2026_06"


@pytest.mark.unit
def test_run_agent_decision_records_system_exit(monkeypatch, tmp_path):
    class ExitingGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, *args, **kwargs):
            raise SystemExit("provider transport aborted")

    monkeypatch.setattr(continuous, "TradingAgentsGraph", ExitingGraph)

    row, state = continuous.run_agent_decision(
        {"ticker": "600519.SS", "name": "", "board": "", "sector": ""},
        "2026-06-01",
        "2026-06-02",
        ["market"],
        tmp_path,
        {
            "llm_provider": "kimi",
            "quick_think_llm": "quick",
            "deep_think_llm": "deep",
        },
        debug=False,
    )

    assert state is None
    assert row.status == "error"
    assert "SystemExit" in row.error


@pytest.mark.unit
def test_run_agent_decision_passes_current_position(monkeypatch, tmp_path):
    captured = {}

    class CapturingGraph:
        def __init__(self, *args, **kwargs):
            captured["config"] = kwargs["config"]

        def propagate(self, *args, **kwargs):
            return {
                "final_trade_decision": "**Rating**: Hold",
                "trader_investment_plan": "**Action**: Hold",
            }, "Hold"

        def save_reports(self, *args, **kwargs):
            return tmp_path / "report.md"

    monkeypatch.setattr(continuous, "TradingAgentsGraph", CapturingGraph)

    row, state = continuous.run_agent_decision(
        {"ticker": "600519.SS", "name": "", "board": "", "sector": ""},
        "2026-06-01",
        "2026-06-02",
        ["market"],
        tmp_path,
        {
            "llm_provider": "deepseek",
            "quick_think_llm": "quick",
            "deep_think_llm": "deep",
        },
        debug=False,
        current_position=1.0,
        opportunity_evidence={"enabled": True, "allow_opportunity": False},
    )

    assert row.status == "ok"
    assert state is not None
    assert captured["config"]["current_position"] == 1.0
    assert captured["config"]["evolution_opportunity_evidence"]["allow_opportunity"] is False


@pytest.mark.unit
def test_run_agent_decision_preserves_keyboard_interrupt(monkeypatch, tmp_path):
    class InterruptingGraph:
        def __init__(self, *args, **kwargs):
            pass

        def propagate(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(continuous, "TradingAgentsGraph", InterruptingGraph)

    with pytest.raises(KeyboardInterrupt):
        continuous.run_agent_decision(
            {"ticker": "600519.SS", "name": "", "board": "", "sector": ""},
            "2026-06-01",
            "2026-06-02",
            ["market"],
            tmp_path,
            {
                "llm_provider": "kimi",
                "quick_think_llm": "quick",
                "deep_think_llm": "deep",
            },
            debug=False,
        )


@pytest.mark.unit
def test_latest_decision_rows_deduplicates_force_reruns():
    old = continuous.DecisionRow(
        ticker="600519.SS",
        name="",
        board="",
        sector="",
        analysis_date="2026-06-01",
        next_date="2026-06-02",
        analysts="market",
        llm_provider="deepseek",
        quick_model="quick",
        deep_model="deep",
        trader_action="Buy",
    )
    new = continuous.DecisionRow(
        ticker="600519.SS",
        name="",
        board="",
        sector="",
        analysis_date="2026-06-01",
        next_date="2026-06-02",
        analysts="market",
        llm_provider="deepseek",
        quick_model="quick",
        deep_model="deep",
        trader_action="Hold",
    )

    latest = continuous.latest_decision_rows([old, new])

    assert len(latest) == 1
    assert latest[0].trader_action == "Hold"
