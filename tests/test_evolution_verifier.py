import csv
import json
from pathlib import Path

import pytest

from tradingagents.evolution.verifier import (
    VerificationThresholds,
    research_gate_thresholds,
    verify_skill_experiment,
    write_verification_artifacts,
)


def _write_result_dir(
    root: Path,
    *,
    returns: list[float],
    actions: list[str],
    positions: list[float],
    stock_returns: list[float] | None = None,
    benchmark_returns: list[float] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stock_returns = stock_returns or returns
    benchmark_returns = benchmark_returns or [0.0 for _ in returns]
    dates = [f"2026-04-{idx + 1:02d}" for idx in range(len(returns))]
    next_dates = [f"2026-04-{idx + 2:02d}" for idx in range(len(returns))]

    with (root / "daily_portfolio.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "analysis_date",
                "next_date",
                "strategy_return",
                "buy_hold_return",
                "benchmark_return",
            ],
        )
        writer.writeheader()
        for idx, value in enumerate(returns):
            writer.writerow(
                {
                    "analysis_date": dates[idx],
                    "next_date": next_dates[idx],
                    "strategy_return": value,
                    "buy_hold_return": stock_returns[idx],
                    "benchmark_return": benchmark_returns[idx],
                }
            )

    with (root / "continuous_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "analysis_date",
                "next_date",
                "status",
                "execution_action",
                "rating",
                "position_before",
                "position_after",
                "strategy_return_next",
                "stock_return_next",
                "benchmark_return_next",
            ],
        )
        writer.writeheader()
        previous = 0.0
        for idx, _value in enumerate(returns):
            position = positions[idx]
            writer.writerow(
                {
                    "ticker": "600519.SS",
                    "analysis_date": dates[idx],
                    "next_date": next_dates[idx],
                    "status": "ok",
                    "execution_action": actions[idx],
                    "rating": "Hold",
                    "position_before": previous,
                    "position_after": position,
                    "strategy_return_next": returns[idx],
                    "stock_return_next": stock_returns[idx],
                    "benchmark_return_next": benchmark_returns[idx],
                }
            )
            previous = position
    return root


def _write_skills(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "skill_id": "opportunity-cash-drag-positive-stock-interval",
            "skill_type": "opportunity",
            "title": "Opportunity: Avoid cash drag",
        }
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_verify_skill_experiment_passes_when_evolved_improves(tmp_path):
    baseline = _write_result_dir(
        tmp_path / "baseline",
        returns=[0.0, 0.0, 0.0],
        actions=["Hold", "Hold", "Hold"],
        positions=[0.0, 0.0, 0.0],
        stock_returns=[0.01, -0.01, 0.0],
        benchmark_returns=[0.0, 0.0, 0.0],
    )
    evolved = _write_result_dir(
        tmp_path / "evolved",
        returns=[0.01, 0.0, 0.0],
        actions=["Buy", "Hold", "Hold"],
        positions=[0.2, 0.2, 0.2],
        stock_returns=[0.01, -0.01, 0.0],
        benchmark_returns=[0.0, 0.0, 0.0],
    )
    skills = _write_skills(tmp_path / "skills" / "candidate_skills.jsonl")

    result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        skills_jsonl=skills,
    )

    assert result.passed
    assert result.accepted_skill_count == 1
    assert result.evolved_metrics.cumulative_return > result.baseline_metrics.cumulative_return
    assert all(gate.passed for gate in result.gates)


@pytest.mark.unit
def test_verify_skill_experiment_rejects_return_regression(tmp_path):
    baseline = _write_result_dir(
        tmp_path / "baseline",
        returns=[0.01, 0.0, 0.0],
        actions=["Buy", "Hold", "Hold"],
        positions=[1.0, 1.0, 1.0],
    )
    evolved = _write_result_dir(
        tmp_path / "evolved",
        returns=[-0.01, 0.0, 0.0],
        actions=["Buy", "Hold", "Hold"],
        positions=[1.0, 1.0, 1.0],
    )

    result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        thresholds=VerificationThresholds(min_return_delta=0.0),
    )

    assert not result.passed
    failed = {gate.name for gate in result.gates if not gate.passed}
    assert "return_not_worse" in failed


@pytest.mark.unit
def test_write_verification_artifacts_accepts_or_rejects_skills(tmp_path):
    baseline = _write_result_dir(
        tmp_path / "baseline",
        returns=[0.0, 0.0],
        actions=["Hold", "Hold"],
        positions=[0.0, 0.0],
    )
    evolved = _write_result_dir(
        tmp_path / "evolved",
        returns=[0.01, 0.0],
        actions=["Buy", "Hold"],
        positions=[0.2, 0.2],
    )
    skills = _write_skills(tmp_path / "skills" / "candidate_skills.jsonl")
    result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        skills_jsonl=skills,
    )

    files = write_verification_artifacts(result, tmp_path / "verification", skills_jsonl=skills)

    assert (tmp_path / "verification" / "skill_verification.json").exists()
    assert (tmp_path / "verification" / "skill_verification.md").exists()
    accepted_path = Path(files["accepted_or_rejected_skills"])
    assert accepted_path.name == "accepted_skills.jsonl"
    assert "opportunity-cash-drag" in accepted_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_research_gate_rejects_zero_activity_zero_return_run(tmp_path):
    baseline = _write_result_dir(
        tmp_path / "baseline",
        returns=[0.0, 0.0, 0.0],
        actions=["Hold", "Hold", "Hold"],
        positions=[0.0, 0.0, 0.0],
        stock_returns=[0.01, 0.0, 0.0],
    )
    evolved = _write_result_dir(
        tmp_path / "evolved",
        returns=[0.0, 0.0, 0.0],
        actions=["Hold", "Hold", "Hold"],
        positions=[0.0, 0.0, 0.0],
        stock_returns=[0.01, 0.0, 0.0],
    )
    skills = _write_skills(tmp_path / "skills" / "accepted_skills.jsonl")

    default_result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        skills_jsonl=skills,
    )
    research_result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        skills_jsonl=skills,
        thresholds=research_gate_thresholds(),
    )

    assert default_result.passed
    assert not research_result.passed
    failed = {gate.name for gate in research_result.gates if not gate.passed}
    assert "return_not_worse" in failed
    assert "active_decision_floor" in failed
    assert "trade_activity_floor" in failed
    assert "decision_change_floor" in failed


@pytest.mark.unit
def test_research_gate_passes_active_positive_skill_change(tmp_path):
    baseline = _write_result_dir(
        tmp_path / "baseline",
        returns=[0.0, 0.0, 0.0],
        actions=["Hold", "Hold", "Hold"],
        positions=[0.0, 0.0, 0.0],
        stock_returns=[0.002, 0.0, 0.0],
    )
    evolved = _write_result_dir(
        tmp_path / "evolved",
        returns=[0.002, 0.0, 0.0],
        actions=["Buy", "Hold", "Hold"],
        positions=[0.2, 0.2, 0.2],
        stock_returns=[0.002, 0.0, 0.0],
    )
    skills = _write_skills(tmp_path / "skills" / "accepted_skills.jsonl")

    result = verify_skill_experiment(
        baseline_dir=baseline,
        evolved_dir=evolved,
        skills_jsonl=skills,
        thresholds=research_gate_thresholds(),
    )

    assert result.passed
    assert result.change_diagnostics.changed_decision_count == 3
    assert result.change_diagnostics.positive_changed_decision_count == 1
    assert result.evolved_diagnostics.active_position_count == 3
