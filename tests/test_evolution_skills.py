import json

import pytest

from tradingagents.evolution.skills import (
    generate_candidate_skills,
    load_candidate_skill_records,
    load_experience_records,
    render_skill_context,
    select_candidate_skills,
    write_skill_artifacts,
)


def _experience(
    *,
    action: str,
    rating: str,
    label: str,
    edge: float,
    ticker: str = "600519.SS",
) -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-04-01",
        "rating": rating,
        "execution_action": action,
        "outcome_label": label,
        "strategy_vs_benchmark": edge,
        "strategy_vs_buy_hold": edge / 2,
        "stock_return_next": edge,
        "outcome_reason": "synthetic reason",
        "reflection": "synthetic reflection",
    }


@pytest.mark.unit
def test_generate_candidate_skills_promotes_successful_pattern():
    experiences = [
        _experience(action="Buy", rating="Overweight", label="success", edge=0.02),
        _experience(action="Buy", rating="Overweight", label="success", edge=0.01),
        _experience(action="Buy", rating="Overweight", label="neutral", edge=0.0),
    ]

    skills = generate_candidate_skills(experiences, min_support=3, promote_success_rate=0.60)

    buy_skill = next(skill for skill in skills if skill.source_dimension == "execution_action")
    assert buy_skill.skill_type == "promote"
    assert buy_skill.success_count == 2
    assert buy_skill.evidence_count == 3
    assert buy_skill.avg_strategy_vs_benchmark == pytest.approx(0.01)


@pytest.mark.unit
def test_generate_candidate_skills_warns_on_underperforming_pattern():
    experiences = [
        _experience(action="Hold", rating="Hold", label="failure", edge=-0.02),
        _experience(action="Hold", rating="Hold", label="failure", edge=-0.01),
        _experience(action="Hold", rating="Hold", label="success", edge=0.01),
    ]

    skills = generate_candidate_skills(experiences, min_support=3, warn_failure_rate=0.50)

    hold_skill = next(skill for skill in skills if skill.source_dimension == "execution_action")
    assert hold_skill.skill_type == "caution"
    assert hold_skill.failure_count == 2
    assert "benchmark-relative" in hold_skill.procedure[-1]


@pytest.mark.unit
def test_skill_artifact_round_trip(tmp_path):
    input_path = tmp_path / "trading_experiences.jsonl"
    records = [
        _experience(action="Sell", rating="Underweight", label="failure", edge=-0.02),
        _experience(action="Sell", rating="Underweight", label="failure", edge=-0.01),
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_experience_records(input_path)
    skills = generate_candidate_skills(loaded, min_support=2)
    manifest = write_skill_artifacts(skills, tmp_path / "skills")

    assert manifest["skill_count"] >= 1
    assert (tmp_path / "skills" / "candidate_skills.jsonl").exists()
    assert (tmp_path / "skills" / "candidate_skills.md").exists()
    assert (tmp_path / "skills" / "skill_manifest.json").exists()
    manifest_file = json.loads((tmp_path / "skills" / "skill_manifest.json").read_text(encoding="utf-8"))
    assert manifest_file["files"]["manifest"].endswith("skill_manifest.json")


@pytest.mark.unit
def test_generate_candidate_skills_adds_cash_drag_opportunity():
    experiences = [
        {
            **_experience(action="Hold", rating="Hold", label="failure", edge=-0.02),
            "position_after": 0.0,
            "stock_return_next": 0.012,
            "benchmark_return_next": 0.014,
            "strategy_vs_buy_hold": -0.012,
        },
        {
            **_experience(action="Sell", rating="Underweight", label="failure", edge=-0.01),
            "position_after": 0.0,
            "stock_return_next": 0.006,
            "benchmark_return_next": 0.011,
            "strategy_vs_buy_hold": -0.006,
        },
    ]

    skills = generate_candidate_skills(
        experiences,
        min_support=2,
        missed_upside_return=0.005,
    )

    opportunity_skills = [skill for skill in skills if skill.skill_type == "opportunity"]
    assert {skill.source_value for skill in opportunity_skills} == {
        "positive_benchmark_interval",
        "positive_stock_interval",
    }
    assert all("cash" in skill.trigger.lower() for skill in opportunity_skills)


@pytest.mark.unit
def test_select_and_render_candidate_skill_context(tmp_path):
    skills_path = tmp_path / "candidate_skills.jsonl"
    records = [
        {
            "skill_id": "caution-rating-hold",
            "title": "Caution: rating=Hold",
            "skill_type": "caution",
            "source_dimension": "rating",
            "source_value": "Hold",
            "trigger": "check benchmark edge",
            "procedure": ["Require a benchmark-relative reason."],
            "evidence_count": 10,
            "avg_strategy_vs_benchmark": -0.01,
        },
        {
            "skill_id": "caution-execution-action-sell",
            "title": "Caution: execution_action=Sell",
            "skill_type": "caution",
            "source_dimension": "execution_action",
            "source_value": "Sell",
            "trigger": "check missed rebound risk",
            "procedure": ["Check upside catalyst."],
            "evidence_count": 5,
            "avg_strategy_vs_benchmark": -0.03,
        },
    ]
    skills_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_candidate_skill_records(skills_path)
    selected = select_candidate_skills(loaded, rating="Hold", max_skills=1)
    context = render_skill_context(selected, max_chars=180)

    assert selected[0]["skill_id"] == "caution-rating-hold"
    assert "Candidate trading skills" in context
    assert len(context) <= 180


@pytest.mark.unit
def test_select_candidate_skills_filters_weak_benchmark_opportunity():
    skills = [
        {
            "skill_id": "caution-rating-hold",
            "skill_type": "caution",
            "source_dimension": "rating",
            "source_value": "Hold",
            "evidence_count": 100,
            "avg_strategy_vs_benchmark": -0.03,
        },
        {
            "skill_id": "opportunity-cash-drag-positive-benchmark-interval",
            "skill_type": "opportunity",
            "source_dimension": "cash_drag",
            "source_value": "positive_benchmark_interval",
            "evidence_count": 20,
            "success_rate": 0.0,
            "failure_rate": 1.0,
            "avg_strategy_vs_benchmark": -0.01,
            "avg_strategy_vs_buy_hold": -0.02,
            "avg_stock_return_next": 0.001,
        },
    ]

    selected = select_candidate_skills(skills, max_skills=2)

    assert [skill["skill_id"] for skill in selected] == ["caution-rating-hold"]


@pytest.mark.unit
def test_select_candidate_skills_keeps_actionable_stock_opportunity():
    skills = [
        {
            "skill_id": "caution-rating-hold",
            "skill_type": "caution",
            "source_dimension": "rating",
            "source_value": "Hold",
            "evidence_count": 100,
            "avg_strategy_vs_benchmark": -0.03,
        },
        {
            "skill_id": "opportunity-cash-drag-positive-stock-interval",
            "skill_type": "opportunity",
            "source_dimension": "cash_drag",
            "source_value": "positive_stock_interval",
            "evidence_count": 12,
            "success_rate": 0.42,
            "failure_rate": 0.58,
            "avg_strategy_vs_benchmark": -0.004,
            "avg_strategy_vs_buy_hold": -0.012,
            "avg_stock_return_next": 0.012,
        },
    ]

    selected = select_candidate_skills(skills, max_skills=2)

    assert selected[0]["skill_id"] == "opportunity-cash-drag-positive-stock-interval"
    assert selected[1]["skill_id"] == "caution-rating-hold"


@pytest.mark.unit
def test_select_candidate_skills_filters_allowed_skill_types():
    skills = [
        {
            "skill_id": "caution-rating-hold",
            "skill_type": "caution",
            "source_dimension": "rating",
            "source_value": "Hold",
            "evidence_count": 100,
            "avg_strategy_vs_benchmark": -0.03,
        },
        {
            "skill_id": "opportunity-cash-drag-positive-stock-interval",
            "skill_type": "opportunity",
            "source_dimension": "cash_drag",
            "source_value": "positive_stock_interval",
            "evidence_count": 20,
            "avg_strategy_vs_benchmark": 0.01,
        },
    ]

    selected = select_candidate_skills(
        skills,
        allowed_skill_types=["opportunity", "promote"],
        max_skills=2,
    )

    assert [skill["skill_id"] for skill in selected] == [
        "opportunity-cash-drag-positive-stock-interval"
    ]
