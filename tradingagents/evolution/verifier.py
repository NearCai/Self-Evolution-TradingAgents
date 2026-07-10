"""Verify whether injected trading skills improve a held-out backtest window.

The verifier is the first gate in a Hermes-style self-evolution loop: candidate
skills are allowed into the next prompt only if an experiment using them is not
worse than the matched baseline under explicit trading constraints.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationThresholds:
    min_return_delta: float = 0.0
    max_drawdown_worsening: float = 0.005
    max_cash_drag_delta: int = 1
    max_turnover_delta: float = 0.25
    cash_up_threshold: float = 0.005


@dataclass(frozen=True)
class PortfolioSnapshot:
    periods: int
    cumulative_return: float
    max_drawdown: float
    sharpe: float | None
    daily_mean_return: float
    daily_volatility: float


@dataclass(frozen=True)
class DecisionDiagnostics:
    decision_count: int
    ok_count: int
    status_counts: dict[str, int]
    execution_action_counts: dict[str, int]
    rating_counts: dict[str, int]
    avg_position_after: float | None
    avg_turnover: float | None
    cash_stock_up_count: int
    cash_benchmark_up_count: int

    @property
    def cash_drag_count(self) -> int:
        return self.cash_stock_up_count + self.cash_benchmark_up_count


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    baseline_value: float | int | str | None
    evolved_value: float | int | str | None
    threshold: float | int | str | None
    message: str


@dataclass(frozen=True)
class SkillVerificationResult:
    baseline_dir: str
    evolved_dir: str
    skills_jsonl: str | None
    skill_count: int
    accepted_skill_count: int
    baseline_metrics: PortfolioSnapshot
    evolved_metrics: PortfolioSnapshot
    baseline_diagnostics: DecisionDiagnostics
    evolved_diagnostics: DecisionDiagnostics
    gates: list[GateResult]
    passed: bool


def verify_skill_experiment(
    *,
    baseline_dir: Path,
    evolved_dir: Path,
    skills_jsonl: Path | None = None,
    thresholds: VerificationThresholds | None = None,
) -> SkillVerificationResult:
    thresholds = thresholds or VerificationThresholds()
    evolved_decisions = _read_decisions(evolved_dir)
    if not evolved_decisions:
        raise ValueError(f"No evolved decisions found in {evolved_dir}")

    keys = _decision_keys(evolved_decisions)
    dates = {date for _, date in keys}
    baseline_decisions = _filter_decisions(_read_decisions(baseline_dir), keys)
    evolved_decisions = _filter_decisions(evolved_decisions, keys)

    baseline_daily = _filter_daily_rows(_read_daily(baseline_dir), dates)
    evolved_daily = _filter_daily_rows(_read_daily(evolved_dir), dates)

    baseline_metrics = _portfolio_snapshot(baseline_daily, "strategy_return")
    evolved_metrics = _portfolio_snapshot(evolved_daily, "strategy_return")
    baseline_diag = _decision_diagnostics(baseline_decisions, thresholds.cash_up_threshold)
    evolved_diag = _decision_diagnostics(evolved_decisions, thresholds.cash_up_threshold)
    skills = _load_skill_records(skills_jsonl) if skills_jsonl else []

    gates = _build_gates(
        baseline_decisions=baseline_decisions,
        evolved_decisions=evolved_decisions,
        baseline_daily=baseline_daily,
        evolved_daily=evolved_daily,
        baseline_metrics=baseline_metrics,
        evolved_metrics=evolved_metrics,
        baseline_diag=baseline_diag,
        evolved_diag=evolved_diag,
        skill_count=len(skills),
        skills_required=skills_jsonl is not None,
        thresholds=thresholds,
    )
    passed = all(gate.passed for gate in gates)
    return SkillVerificationResult(
        baseline_dir=str(baseline_dir),
        evolved_dir=str(evolved_dir),
        skills_jsonl=str(skills_jsonl) if skills_jsonl else None,
        skill_count=len(skills),
        accepted_skill_count=len(skills) if passed else 0,
        baseline_metrics=baseline_metrics,
        evolved_metrics=evolved_metrics,
        baseline_diagnostics=baseline_diag,
        evolved_diagnostics=evolved_diag,
        gates=gates,
        passed=passed,
    )


def write_verification_artifacts(
    result: SkillVerificationResult,
    output_dir: Path,
    *,
    skills_jsonl: Path | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "skill_verification.json"
    md_path = output_dir / "skill_verification.md"
    json_path.write_text(json.dumps(_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")

    files = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    if skills_jsonl:
        records = _load_skill_records(skills_jsonl)
        target = output_dir / ("accepted_skills.jsonl" if result.passed else "rejected_skills.jsonl")
        with target.open("w", encoding="utf-8") as f:
            for record in records:
                if not result.passed:
                    record = {
                        **record,
                        "rejection_reason": _failed_gate_summary(result.gates),
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        files["accepted_or_rejected_skills"] = str(target)
    return files


def _build_gates(
    *,
    baseline_decisions: list[dict[str, str]],
    evolved_decisions: list[dict[str, str]],
    baseline_daily: list[dict[str, str]],
    evolved_daily: list[dict[str, str]],
    baseline_metrics: PortfolioSnapshot,
    evolved_metrics: PortfolioSnapshot,
    baseline_diag: DecisionDiagnostics,
    evolved_diag: DecisionDiagnostics,
    skill_count: int,
    skills_required: bool,
    thresholds: VerificationThresholds,
) -> list[GateResult]:
    gates: list[GateResult] = []
    gates.append(
        GateResult(
            name="completed_runs",
            passed=evolved_diag.decision_count > 0 and evolved_diag.ok_count == evolved_diag.decision_count,
            baseline_value=None,
            evolved_value=f"{evolved_diag.ok_count}/{evolved_diag.decision_count}",
            threshold="all ok",
            message="All evolved-window decisions must complete successfully.",
        )
    )
    gates.append(
        GateResult(
            name="matched_window",
            passed=len(baseline_decisions) == len(evolved_decisions)
            and len(baseline_daily) == len(evolved_daily)
            and len(evolved_daily) > 0,
            baseline_value=f"{len(baseline_decisions)} decisions, {len(baseline_daily)} days",
            evolved_value=f"{len(evolved_decisions)} decisions, {len(evolved_daily)} days",
            threshold="same dates and tickers",
            message="Baseline and evolved runs must overlap on the same evaluation window.",
        )
    )
    if skills_required:
        gates.append(
            GateResult(
                name="skills_present",
                passed=skill_count > 0,
                baseline_value=None,
                evolved_value=skill_count,
                threshold=">0",
                message="A skill file was supplied and must contain at least one candidate skill.",
            )
        )
    return_delta = evolved_metrics.cumulative_return - baseline_metrics.cumulative_return
    gates.append(
        GateResult(
            name="return_not_worse",
            passed=return_delta >= thresholds.min_return_delta,
            baseline_value=baseline_metrics.cumulative_return,
            evolved_value=evolved_metrics.cumulative_return,
            threshold=thresholds.min_return_delta,
            message="Evolved strategy cumulative return must not trail the matched baseline.",
        )
    )
    drawdown_worsening = baseline_metrics.max_drawdown - evolved_metrics.max_drawdown
    gates.append(
        GateResult(
            name="drawdown_guard",
            passed=drawdown_worsening <= thresholds.max_drawdown_worsening,
            baseline_value=baseline_metrics.max_drawdown,
            evolved_value=evolved_metrics.max_drawdown,
            threshold=thresholds.max_drawdown_worsening,
            message="Evolved strategy may not worsen max drawdown beyond the tolerance.",
        )
    )
    cash_drag_delta = evolved_diag.cash_drag_count - baseline_diag.cash_drag_count
    gates.append(
        GateResult(
            name="cash_drag_guard",
            passed=cash_drag_delta <= thresholds.max_cash_drag_delta,
            baseline_value=baseline_diag.cash_drag_count,
            evolved_value=evolved_diag.cash_drag_count,
            threshold=thresholds.max_cash_drag_delta,
            message="Evolved strategy should not create materially more cash-drag intervals.",
        )
    )
    turnover_delta = _none_safe_subtract(evolved_diag.avg_turnover, baseline_diag.avg_turnover)
    gates.append(
        GateResult(
            name="turnover_guard",
            passed=turnover_delta is None or turnover_delta <= thresholds.max_turnover_delta,
            baseline_value=baseline_diag.avg_turnover,
            evolved_value=evolved_diag.avg_turnover,
            threshold=thresholds.max_turnover_delta,
            message="Evolved strategy should not increase turnover beyond the tolerance.",
        )
    )
    return gates


def _read_decisions(result_dir: Path) -> list[dict[str, str]]:
    path = result_dir / "continuous_decisions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing continuous decisions file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_daily(result_dir: Path) -> list[dict[str, str]]:
    path = result_dir / "daily_portfolio.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing daily portfolio file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _decision_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {
        (row.get("ticker", ""), row.get("analysis_date", ""))
        for row in rows
        if row.get("ticker") and row.get("analysis_date")
    }


def _filter_decisions(
    rows: list[dict[str, str]],
    keys: set[tuple[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if (row.get("ticker", ""), row.get("analysis_date", "")) in keys
    ]


def _filter_daily_rows(
    rows: list[dict[str, str]],
    dates: set[str],
) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row.get("analysis_date") in dates],
        key=lambda row: row.get("analysis_date", ""),
    )


def _portfolio_snapshot(rows: list[dict[str, str]], return_column: str) -> PortfolioSnapshot:
    returns = [_to_float(row.get(return_column)) or 0.0 for row in rows]
    equity = 1.0
    curve: list[float] = []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    cumulative_return = equity - 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    mean = sum(returns) / len(returns) if returns else 0.0
    volatility = _sample_std(returns)
    sharpe = (mean / volatility * (252**0.5)) if volatility else None
    return PortfolioSnapshot(
        periods=len(returns),
        cumulative_return=cumulative_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        daily_mean_return=mean,
        daily_volatility=volatility,
    )


def _decision_diagnostics(
    rows: list[dict[str, str]],
    cash_up_threshold: float,
) -> DecisionDiagnostics:
    positions = [_to_float(row.get("position_after")) for row in rows]
    positions = [value for value in positions if value is not None]
    turnovers = []
    for row in rows:
        before = _to_float(row.get("position_before"))
        after = _to_float(row.get("position_after"))
        if before is not None and after is not None:
            turnovers.append(abs(after - before))
    cash_stock_up = 0
    cash_benchmark_up = 0
    for row in rows:
        position = _to_float(row.get("position_after")) or 0.0
        if position != 0.0:
            continue
        if (_to_float(row.get("stock_return_next")) or 0.0) >= cash_up_threshold:
            cash_stock_up += 1
        if (_to_float(row.get("benchmark_return_next")) or 0.0) >= cash_up_threshold:
            cash_benchmark_up += 1
    return DecisionDiagnostics(
        decision_count=len(rows),
        ok_count=sum(1 for row in rows if row.get("status") == "ok"),
        status_counts=dict(sorted(Counter(row.get("status", "") for row in rows).items())),
        execution_action_counts=dict(
            sorted(Counter(row.get("execution_action", "") for row in rows).items())
        ),
        rating_counts=dict(sorted(Counter(row.get("rating", "") for row in rows).items())),
        avg_position_after=(sum(positions) / len(positions)) if positions else None,
        avg_turnover=(sum(turnovers) / len(turnovers)) if turnovers else None,
        cash_stock_up_count=cash_stock_up,
        cash_benchmark_up_count=cash_benchmark_up,
    )


def _load_skill_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Missing skills JSONL: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _render_markdown(result: SkillVerificationResult) -> str:
    status = "PASSED" if result.passed else "FAILED"
    lines = [
        "# Trading Skill Verification",
        "",
        f"Status: **{status}**",
        "",
        f"- Baseline: `{result.baseline_dir}`",
        f"- Evolved: `{result.evolved_dir}`",
        f"- Skills: `{result.skills_jsonl or 'n/a'}`",
        f"- Skill count: {result.skill_count}",
        f"- Accepted skill count: {result.accepted_skill_count}",
        "",
        "## Portfolio Metrics",
        "",
        "| Run | Periods | CR | Sharpe | MDD | Avg Daily Return |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("Baseline", result.baseline_metrics),
        _metric_row("Evolved", result.evolved_metrics),
        "",
        "## Decision Diagnostics",
        "",
        "| Run | Decisions | OK | Avg Position | Avg Turnover | Cash Drag | Actions |",
        "|---|---:|---:|---:|---:|---:|---|",
        _diagnostic_row("Baseline", result.baseline_diagnostics),
        _diagnostic_row("Evolved", result.evolved_diagnostics),
        "",
        "## Gates",
        "",
        "| Gate | Pass | Baseline | Evolved | Threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    for gate in result.gates:
        lines.append(
            f"| {gate.name} | {'yes' if gate.passed else 'no'} | "
            f"{_format_value(gate.baseline_value)} | {_format_value(gate.evolved_value)} | "
            f"{_format_value(gate.threshold)} |"
        )
    lines.append("")
    failed = [gate.message for gate in result.gates if not gate.passed]
    if failed:
        lines.extend(["Failed gate notes:", *[f"- {message}" for message in failed], ""])
    return "\n".join(lines)


def _metric_row(name: str, snapshot: PortfolioSnapshot) -> str:
    return (
        f"| {name} | {snapshot.periods} | {_format_pct(snapshot.cumulative_return)} | "
        f"{_format_float(snapshot.sharpe)} | {_format_pct(snapshot.max_drawdown)} | "
        f"{_format_pct(snapshot.daily_mean_return)} |"
    )


def _diagnostic_row(name: str, diag: DecisionDiagnostics) -> str:
    actions = ", ".join(f"{key}:{value}" for key, value in diag.execution_action_counts.items())
    return (
        f"| {name} | {diag.decision_count} | {diag.ok_count} | "
        f"{_format_float(diag.avg_position_after)} | {_format_float(diag.avg_turnover)} | "
        f"{diag.cash_drag_count} | {actions} |"
    )


def _failed_gate_summary(gates: list[GateResult]) -> str:
    failed = [gate.name for gate in gates if not gate.passed]
    return ", ".join(failed) if failed else ""


def _to_dict(result: SkillVerificationResult) -> dict[str, Any]:
    return asdict(result)


def _none_safe_subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_pct(value: float | int | str | None) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2%}"


def _format_float(value: float | int | str | None) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}"


def _format_value(value: float | int | str | None) -> str:
    if isinstance(value, float):
        if abs(value) <= 1.0:
            return _format_pct(value)
        return _format_float(value)
    if value is None:
        return "n/a"
    return str(value)
