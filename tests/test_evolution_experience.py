import csv
import json

import pytest

from tradingagents.evolution.experience import (
    build_experiences,
    classify_outcome,
    write_experience_artifacts,
)


@pytest.mark.unit
def test_classify_outcome_uses_benchmark_edge():
    label, reason = classify_outcome(
        action="Buy",
        position_after=1.0,
        stock_return_next=0.02,
        strategy_return_next=0.02,
        benchmark_return_next=0.01,
    )

    assert label == "success"
    assert "beat benchmark" in reason


@pytest.mark.unit
def test_classify_outcome_flags_missed_positive_stock_interval():
    label, reason = classify_outcome(
        action="Sell",
        position_after=0.0,
        stock_return_next=0.003,
        strategy_return_next=0.0,
        benchmark_return_next=0.001,
    )

    assert label == "failure"
    assert "missed" in reason


@pytest.mark.unit
def test_build_experiences_reads_decisions_state_and_memory(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "market_report": "market " * 20,
                "fundamentals_report": "fundamentals report",
                "investment_plan": "investment plan",
                "final_trade_decision": "final decision",
            }
        ),
        encoding="utf-8",
    )

    decisions_path = tmp_path / "continuous_decisions.csv"
    rows = [
        {
            "ticker": "600519.SS",
            "analysis_date": "2026-04-01",
            "next_date": "2026-04-02",
            "analysts": "market,fundamentals",
            "llm_provider": "deepseek",
            "quick_model": "quick",
            "deep_model": "deep",
            "rating": "Underweight",
            "trader_action": "Sell",
            "execution_action": "Sell",
            "decision_source": "pm-rating",
            "position_before": "1.0",
            "position_after": "0.0",
            "stock_return_next": "-0.02",
            "strategy_return_next": "0.0",
            "benchmark": "000001.SS",
            "benchmark_return_next": "-0.01",
            "state_path": str(state_path),
            "report_path": str(tmp_path / "reports"),
            "status": "ok",
        }
    ]
    with decisions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    memory_dir = tmp_path / "_memory"
    memory_dir.mkdir()
    (memory_dir / "continuous_memory.md").write_text(
        "[2026-04-01 | 600519.SS | Underweight | -2.0% | -1.0% | 5d]\n\n"
        "DECISION:\nold decision\n\n"
        "REFLECTION:\nAvoid weak rebound signals next time.\n\n"
        "<!-- ENTRY_END -->\n",
        encoding="utf-8",
    )

    experiences = build_experiences(tmp_path, max_text_chars=40)

    assert len(experiences) == 1
    exp = experiences[0]
    assert exp.ticker == "600519.SS"
    assert exp.alpha_next == pytest.approx(-0.01)
    assert exp.outcome_label == "success"
    assert exp.reflection == "Avoid weak rebound signals next time."
    assert exp.market_report.endswith("...[truncated]")


@pytest.mark.unit
def test_write_experience_artifacts(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    decisions_path = tmp_path / "continuous_decisions.csv"
    rows = [
        {
            "ticker": "000333.SZ",
            "analysis_date": "2026-04-01",
            "next_date": "2026-04-02",
            "analysts": "market",
            "llm_provider": "deepseek",
            "quick_model": "quick",
            "deep_model": "deep",
            "rating": "Hold",
            "trader_action": "Hold",
            "execution_action": "Hold",
            "decision_source": "pm-rating",
            "position_before": "0.0",
            "position_after": "0.0",
            "stock_return_next": "0.0",
            "strategy_return_next": "0.0",
            "benchmark": "399001.SZ",
            "benchmark_return_next": "0.0",
            "state_path": str(state_path),
            "report_path": "",
            "status": "ok",
        }
    ]
    with decisions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    experiences = build_experiences(tmp_path)
    manifest = write_experience_artifacts(experiences, tmp_path / "out")

    assert manifest["experience_count"] == 1
    assert (tmp_path / "out" / "trading_experiences.jsonl").exists()
    assert (tmp_path / "out" / "experience_summary.csv").exists()
    assert (tmp_path / "out" / "experience_manifest.json").exists()
