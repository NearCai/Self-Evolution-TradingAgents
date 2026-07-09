"""Build structured trading experiences from continuous backtest outputs.

This module is the first offline harness piece for skill self-evolution.  It
turns a completed backtest run into compact, replayable experience records that
later steps can distill into candidate trading skills.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.memory import TradingMemoryLog

SUCCESS_MARGIN = 0.0
FAILURE_MARGIN = -0.005


@dataclass
class TradingExperience:
    ticker: str
    analysis_date: str
    next_date: str
    analysts: str
    llm_provider: str
    quick_model: str
    deep_model: str
    rating: str | None
    trader_action: str | None
    execution_action: str | None
    decision_source: str | None
    position_before: float | None
    position_after: float | None
    stock_return_next: float | None
    strategy_return_next: float | None
    benchmark: str | None
    benchmark_return_next: float | None
    alpha_next: float | None
    strategy_vs_benchmark: float | None
    strategy_vs_buy_hold: float | None
    outcome_label: str
    outcome_reason: str
    market_report: str
    fundamentals_report: str
    investment_plan: str
    final_trade_decision: str
    reflection: str
    state_path: str | None
    report_path: str | None


def read_decision_rows(result_dir: Path) -> list[dict[str, str]]:
    decisions_path = result_dir / "continuous_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing continuous decisions file: {decisions_path}")
    with decisions_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_memory_reflections(result_dir: Path) -> dict[tuple[str, str], str]:
    memory_path = result_dir / "_memory" / "continuous_memory.md"
    if not memory_path.exists():
        return {}
    memory = TradingMemoryLog({"memory_log_path": str(memory_path)})
    reflections: dict[tuple[str, str], str] = {}
    for entry in memory.load_entries():
        reflection = _clean_memory_text(entry.get("reflection") or "")
        if reflection:
            reflections[(entry.get("ticker", ""), entry.get("date", ""))] = reflection
    return reflections


def build_experiences(
    result_dir: Path,
    *,
    max_text_chars: int = 1200,
    require_ok: bool = True,
) -> list[TradingExperience]:
    rows = read_decision_rows(result_dir)
    reflections = load_memory_reflections(result_dir)
    experiences: list[TradingExperience] = []
    for row in rows:
        if require_ok and row.get("status") != "ok":
            continue
        state = _load_state(row.get("state_path"), result_dir)
        alpha_next = _subtract(row.get("stock_return_next"), row.get("benchmark_return_next"))
        strategy_vs_benchmark = _subtract(
            row.get("strategy_return_next"),
            row.get("benchmark_return_next"),
        )
        strategy_vs_buy_hold = _subtract(row.get("strategy_return_next"), row.get("stock_return_next"))
        outcome_label, outcome_reason = classify_outcome(
            action=row.get("execution_action"),
            position_after=_to_float(row.get("position_after")),
            stock_return_next=_to_float(row.get("stock_return_next")),
            strategy_return_next=_to_float(row.get("strategy_return_next")),
            benchmark_return_next=_to_float(row.get("benchmark_return_next")),
        )
        key = (row.get("ticker", ""), row.get("analysis_date", ""))
        experiences.append(
            TradingExperience(
                ticker=row.get("ticker", ""),
                analysis_date=row.get("analysis_date", ""),
                next_date=row.get("next_date", ""),
                analysts=row.get("analysts", ""),
                llm_provider=row.get("llm_provider", ""),
                quick_model=row.get("quick_model", ""),
                deep_model=row.get("deep_model", ""),
                rating=_none_if_blank(row.get("rating")),
                trader_action=_none_if_blank(row.get("trader_action")),
                execution_action=_none_if_blank(row.get("execution_action")),
                decision_source=_none_if_blank(row.get("decision_source")),
                position_before=_to_float(row.get("position_before")),
                position_after=_to_float(row.get("position_after")),
                stock_return_next=_to_float(row.get("stock_return_next")),
                strategy_return_next=_to_float(row.get("strategy_return_next")),
                benchmark=_none_if_blank(row.get("benchmark")),
                benchmark_return_next=_to_float(row.get("benchmark_return_next")),
                alpha_next=alpha_next,
                strategy_vs_benchmark=strategy_vs_benchmark,
                strategy_vs_buy_hold=strategy_vs_buy_hold,
                outcome_label=outcome_label,
                outcome_reason=outcome_reason,
                market_report=_truncate(str(state.get("market_report", "")), max_text_chars),
                fundamentals_report=_truncate(
                    str(state.get("fundamentals_report", "")),
                    max_text_chars,
                ),
                investment_plan=_truncate(str(state.get("investment_plan", "")), max_text_chars),
                final_trade_decision=_truncate(
                    str(state.get("final_trade_decision", "")),
                    max_text_chars,
                ),
                reflection=_truncate(reflections.get(key, ""), max_text_chars),
                state_path=_none_if_blank(row.get("state_path")),
                report_path=_none_if_blank(row.get("report_path")),
            )
        )
    return experiences


def classify_outcome(
    *,
    action: str | None,
    position_after: float | None,
    stock_return_next: float | None,
    strategy_return_next: float | None,
    benchmark_return_next: float | None,
) -> tuple[str, str]:
    if strategy_return_next is None or benchmark_return_next is None:
        return "unknown", "missing realized strategy or benchmark return"

    edge = strategy_return_next - benchmark_return_next
    action_name = (action or "").lower()
    exposure = abs(position_after or 0.0)

    if edge >= SUCCESS_MARGIN:
        return "success", f"strategy beat benchmark by {edge:+.2%} over the next interval"
    if edge <= FAILURE_MARGIN:
        return "failure", f"strategy lagged benchmark by {edge:+.2%} over the next interval"

    if action_name == "sell" and stock_return_next is not None and stock_return_next < 0:
        return "success", "cash/underweight avoided a negative stock interval"
    if exposure == 0.0 and stock_return_next is not None and stock_return_next > 0:
        return "failure", "cash/underweight missed a positive stock interval"
    return "neutral", f"strategy was close to benchmark ({edge:+.2%})"


def write_experience_artifacts(
    experiences: list[TradingExperience],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(item) for item in experiences]

    jsonl_path = output_dir / "trading_experiences.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = output_dir / "experience_summary.csv"
    summary_fields = [
        "ticker",
        "analysis_date",
        "next_date",
        "analysts",
        "rating",
        "execution_action",
        "position_after",
        "stock_return_next",
        "strategy_return_next",
        "benchmark_return_next",
        "strategy_vs_benchmark",
        "strategy_vs_buy_hold",
        "outcome_label",
        "outcome_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in summary_fields})

    label_counts = Counter(item.outcome_label for item in experiences)
    ticker_counts = Counter(item.ticker for item in experiences)
    manifest = {
        "experience_count": len(experiences),
        "label_counts": dict(sorted(label_counts.items())),
        "ticker_counts": dict(sorted(ticker_counts.items())),
        "files": {
            "jsonl": str(jsonl_path),
            "summary_csv": str(csv_path),
        },
    }
    manifest_path = output_dir / "experience_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    return manifest


def _load_state(raw_path: str | None, result_dir: Path) -> dict[str, Any]:
    path_text = (raw_path or "").strip()
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_absolute():
        path = result_dir / path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _to_float(value: str | float | int | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _subtract(left: str | None, right: str | None) -> float | None:
    left_float = _to_float(left)
    right_float = _to_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _clean_memory_text(text: str) -> str:
    return text.split("<!-- ENTRY_END -->", 1)[0].strip()


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
