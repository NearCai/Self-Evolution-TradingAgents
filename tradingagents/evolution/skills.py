"""Generate candidate trading skills from structured experiences.

This module keeps the first self-evolution step offline and deterministic:
completed backtest experiences are distilled into evidence-backed candidate
skills, but the live agent prompt is not changed yet.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CandidateTradingSkill:
    skill_id: str
    title: str
    skill_type: str
    source_dimension: str
    source_value: str
    trigger: str
    procedure: list[str]
    evidence_count: int
    success_count: int
    failure_count: int
    neutral_count: int
    success_rate: float
    failure_rate: float
    avg_strategy_vs_benchmark: float | None
    avg_strategy_vs_buy_hold: float | None
    avg_stock_return_next: float | None
    tickers: list[str]
    example_experiences: list[dict[str, Any]]


def load_experience_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing experience JSONL: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_candidate_skills(
    experiences: list[dict[str, Any]],
    *,
    min_support: int = 5,
    promote_success_rate: float = 0.55,
    warn_failure_rate: float = 0.50,
    max_examples: int = 3,
) -> list[CandidateTradingSkill]:
    skills: list[CandidateTradingSkill] = []
    for dimension in ("execution_action", "rating"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for exp in experiences:
            value = _clean_value(exp.get(dimension))
            if value:
                grouped[value].append(exp)

        for value, group in sorted(grouped.items()):
            if len(group) < min_support:
                continue
            summary = _summarize_group(group)
            skill_type = _classify_skill_type(
                summary["success_rate"],
                summary["failure_rate"],
                summary["avg_strategy_vs_benchmark"],
                promote_success_rate,
                warn_failure_rate,
            )
            if skill_type is None:
                continue
            skills.append(
                CandidateTradingSkill(
                    skill_id=_skill_id(skill_type, dimension, value),
                    title=_title(skill_type, dimension, value),
                    skill_type=skill_type,
                    source_dimension=dimension,
                    source_value=value,
                    trigger=_trigger(skill_type, dimension, value),
                    procedure=_procedure(skill_type, dimension, value),
                    evidence_count=len(group),
                    success_count=summary["labels"].get("success", 0),
                    failure_count=summary["labels"].get("failure", 0),
                    neutral_count=summary["labels"].get("neutral", 0),
                    success_rate=summary["success_rate"],
                    failure_rate=summary["failure_rate"],
                    avg_strategy_vs_benchmark=summary["avg_strategy_vs_benchmark"],
                    avg_strategy_vs_buy_hold=summary["avg_strategy_vs_buy_hold"],
                    avg_stock_return_next=summary["avg_stock_return_next"],
                    tickers=summary["tickers"],
                    example_experiences=_select_examples(group, skill_type, max_examples),
                )
            )
    return skills


def write_skill_artifacts(
    skills: list[CandidateTradingSkill],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(skill) for skill in skills]

    jsonl_path = output_dir / "candidate_skills.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    markdown_path = output_dir / "candidate_skills.md"
    markdown_path.write_text(_render_markdown(skills), encoding="utf-8")

    type_counts = Counter(skill.skill_type for skill in skills)
    manifest = {
        "skill_count": len(skills),
        "type_counts": dict(sorted(type_counts.items())),
        "files": {
            "jsonl": str(jsonl_path),
            "markdown": str(markdown_path),
        },
    }
    manifest_path = output_dir / "skill_manifest.json"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(_clean_value(item.get("outcome_label")) for item in group)
    evidence_count = len(group)
    return {
        "labels": labels,
        "success_rate": labels.get("success", 0) / evidence_count,
        "failure_rate": labels.get("failure", 0) / evidence_count,
        "avg_strategy_vs_benchmark": _average(item.get("strategy_vs_benchmark") for item in group),
        "avg_strategy_vs_buy_hold": _average(item.get("strategy_vs_buy_hold") for item in group),
        "avg_stock_return_next": _average(item.get("stock_return_next") for item in group),
        "tickers": sorted({_clean_value(item.get("ticker")) for item in group if _clean_value(item.get("ticker"))}),
    }


def _classify_skill_type(
    success_rate: float,
    failure_rate: float,
    avg_edge: float | None,
    promote_success_rate: float,
    warn_failure_rate: float,
) -> str | None:
    if success_rate >= promote_success_rate and (avg_edge is None or avg_edge >= 0):
        return "promote"
    if failure_rate >= warn_failure_rate or (avg_edge is not None and avg_edge < 0):
        return "caution"
    return None


def _select_examples(
    group: list[dict[str, Any]],
    skill_type: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    target_label = "success" if skill_type == "promote" else "failure"
    ranked = sorted(
        (item for item in group if item.get("outcome_label") == target_label),
        key=lambda item: abs(_to_float(item.get("strategy_vs_benchmark")) or 0.0),
        reverse=True,
    )
    examples = ranked[:max_examples]
    return [
        {
            "ticker": item.get("ticker"),
            "analysis_date": item.get("analysis_date"),
            "rating": item.get("rating"),
            "execution_action": item.get("execution_action"),
            "strategy_vs_benchmark": item.get("strategy_vs_benchmark"),
            "strategy_vs_buy_hold": item.get("strategy_vs_buy_hold"),
            "outcome_reason": item.get("outcome_reason"),
            "reflection": _shorten(str(item.get("reflection") or ""), 400),
        }
        for item in examples
    ]


def _trigger(skill_type: str, dimension: str, value: str) -> str:
    if skill_type == "promote":
        return (
            f"When the agent is about to choose {dimension}={value}, use this pattern "
            "as positive evidence only if the current reports match the cited examples."
        )
    return (
        f"When the agent is about to choose {dimension}={value}, require an explicit "
        "benchmark-relative reason and a downside/upside catalyst check before acting."
    )


def _procedure(skill_type: str, dimension: str, value: str) -> list[str]:
    if skill_type == "promote":
        return [
            "Compare the current market and fundamentals reports with the evidence examples.",
            "Check whether the expected next-interval result is benchmark-relative, not only stock-relative.",
            f"If the setup still supports {dimension}={value}, keep the decision and record the reason.",
        ]
    return [
        "Treat the historical pattern as a warning rather than a hard ban.",
        "Look for missed rebound risk, benchmark strength, and whether cash/long exposure is being overused.",
        f"Only keep {dimension}={value} when the report names a concrete catalyst and a benchmark-relative edge.",
    ]


def _title(skill_type: str, dimension: str, value: str) -> str:
    prefix = "Promote" if skill_type == "promote" else "Caution"
    return f"{prefix}: {dimension}={value}"


def _skill_id(skill_type: str, dimension: str, value: str) -> str:
    safe_value = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return f"{skill_type}-{dimension.replace('_', '-')}-{safe_value}"


def _render_markdown(skills: list[CandidateTradingSkill]) -> str:
    lines = ["# Candidate Trading Skills", ""]
    if not skills:
        lines.append("No candidate skills met the support and performance thresholds.")
        lines.append("")
        return "\n".join(lines)

    for skill in skills:
        lines.extend(
            [
                f"## {skill.title}",
                "",
                f"- Skill id: `{skill.skill_id}`",
                f"- Type: `{skill.skill_type}`",
                f"- Source: `{skill.source_dimension}={skill.source_value}`",
                f"- Evidence: {skill.evidence_count} cases, success {skill.success_count}, "
                f"failure {skill.failure_count}, neutral {skill.neutral_count}",
                f"- Success rate: {skill.success_rate:.1%}",
                f"- Failure rate: {skill.failure_rate:.1%}",
                f"- Avg strategy vs benchmark: {_format_pct(skill.avg_strategy_vs_benchmark)}",
                f"- Avg strategy vs buy-hold: {_format_pct(skill.avg_strategy_vs_buy_hold)}",
                f"- Tickers: {', '.join(skill.tickers)}",
                "",
                f"Trigger: {skill.trigger}",
                "",
                "Procedure:",
            ]
        )
        lines.extend(f"1. {step}" for step in skill.procedure)
        if skill.example_experiences:
            lines.extend(["", "Evidence examples:"])
            for example in skill.example_experiences:
                lines.append(
                    f"- {example['ticker']} {example['analysis_date']}: "
                    f"{_format_pct(example['strategy_vs_benchmark'])} vs benchmark; "
                    f"{example['outcome_reason']}"
                )
        lines.append("")
    return "\n".join(lines)


def _average(values: Any) -> float | None:
    parsed = [_to_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    if not parsed:
        return None
    return sum(parsed) / len(parsed)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_value(value: Any) -> str:
    return str(value or "").strip()


def _shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_pct(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.2%}"
