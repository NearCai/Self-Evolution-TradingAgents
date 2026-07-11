"""Verify a skill-injected trading run against a matched baseline window."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.evolution.verifier import (  # noqa: E402
    VerificationThresholds,
    research_gate_thresholds,
    verify_skill_experiment,
    write_verification_artifacts,
)

DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
)
DEFAULT_EVOLVED_DIR = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm"
)
DEFAULT_SKILLS_JSONL = DEFAULT_BASELINE_DIR / "evolution" / "skills" / "candidate_skills.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-preset",
        choices=["default", "research"],
        default="default",
        help="Verifier preset. 'research' rejects zero-activity self-evolution runs.",
    )
    parser.add_argument(
        "--research-gate",
        action="store_true",
        help="Shortcut for --gate-preset research.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Baseline continuous backtest result directory.",
    )
    parser.add_argument(
        "--evolved-dir",
        default=str(DEFAULT_EVOLVED_DIR),
        help="Skill-injected continuous backtest result directory.",
    )
    parser.add_argument(
        "--skills-jsonl",
        default=str(DEFAULT_SKILLS_JSONL),
        help="Candidate skills JSONL. Use an empty string to skip skill artifact copying.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <evolved-dir>/skill_verification.",
    )
    parser.add_argument(
        "--min-return-delta",
        type=float,
        default=None,
        help="Minimum evolved cumulative-return improvement over the matched baseline.",
    )
    parser.add_argument(
        "--max-drawdown-worsening",
        type=float,
        default=None,
        help="Allowed max-drawdown worsening, absolute return units.",
    )
    parser.add_argument(
        "--max-cash-drag-delta",
        type=int,
        default=None,
        help="Allowed increase in cash-drag intervals.",
    )
    parser.add_argument(
        "--max-turnover-delta",
        type=float,
        default=None,
        help="Allowed increase in average per-decision turnover.",
    )
    parser.add_argument(
        "--cash-up-threshold",
        type=float,
        default=None,
        help="Positive next-interval return threshold used to count cash-drag cases.",
    )
    parser.add_argument(
        "--min-avg-position-after",
        type=float,
        default=None,
        help="Minimum average post-decision position. Disabled unless set or research preset is used.",
    )
    parser.add_argument(
        "--min-active-decision-ratio",
        type=float,
        default=None,
        help="Minimum ratio of decisions with non-zero position.",
    )
    parser.add_argument(
        "--min-trade-count",
        type=int,
        default=None,
        help="Minimum count of effective position changes.",
    )
    parser.add_argument(
        "--min-decision-change-count",
        type=int,
        default=None,
        help="Minimum decisions changed versus the matched baseline.",
    )
    parser.add_argument(
        "--min-positive-changed-decision-count",
        type=int,
        default=None,
        help="Minimum changed decisions with positive next-interval return contribution.",
    )
    parser.add_argument(
        "--min-changed-return-delta",
        type=float,
        default=None,
        help="Minimum aggregate return delta from skill-induced changed decisions.",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    evolved_dir = Path(args.evolved_dir)
    skills_jsonl = Path(args.skills_jsonl) if args.skills_jsonl else None
    output_dir = Path(args.output_dir) if args.output_dir else evolved_dir / "skill_verification"
    gate_preset = "research" if args.research_gate else args.gate_preset
    thresholds = _build_thresholds(args, gate_preset)
    result = verify_skill_experiment(
        baseline_dir=baseline_dir,
        evolved_dir=evolved_dir,
        skills_jsonl=skills_jsonl,
        thresholds=thresholds,
    )
    files = write_verification_artifacts(result, output_dir, skills_jsonl=skills_jsonl)

    print("Trading skill verifier")
    print("Status:", "PASSED" if result.passed else "FAILED")
    print("Gate preset:", gate_preset)
    print("Baseline:", baseline_dir)
    print("Evolved:", evolved_dir)
    print("Skills:", skills_jsonl or "n/a")
    print("Accepted skills:", result.accepted_skill_count)
    print(
        "Return delta:",
        f"{result.evolved_metrics.cumulative_return - result.baseline_metrics.cumulative_return:+.2%}",
    )
    print("Baseline CR:", f"{result.baseline_metrics.cumulative_return:+.2%}")
    print("Evolved CR:", f"{result.evolved_metrics.cumulative_return:+.2%}")
    print("Changed decisions:", result.change_diagnostics.changed_decision_count)
    print("Positive changed decisions:", result.change_diagnostics.positive_changed_decision_count)
    print(
        "Changed-decision return delta:",
        f"{result.change_diagnostics.changed_return_delta_sum:+.2%}",
    )
    for gate in result.gates:
        print(f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: {gate.message}")
    for path in files.values():
        print("Wrote:", path)
    return 0 if result.passed else 2


def _build_thresholds(args: argparse.Namespace, gate_preset: str) -> VerificationThresholds:
    base = research_gate_thresholds() if gate_preset == "research" else VerificationThresholds()
    return replace(
        base,
        min_return_delta=_coalesce(args.min_return_delta, base.min_return_delta),
        max_drawdown_worsening=_coalesce(
            args.max_drawdown_worsening,
            base.max_drawdown_worsening,
        ),
        max_cash_drag_delta=_coalesce(args.max_cash_drag_delta, base.max_cash_drag_delta),
        max_turnover_delta=_coalesce(args.max_turnover_delta, base.max_turnover_delta),
        cash_up_threshold=_coalesce(args.cash_up_threshold, base.cash_up_threshold),
        min_avg_position_after=_coalesce(
            args.min_avg_position_after,
            base.min_avg_position_after,
        ),
        min_active_decision_ratio=_coalesce(
            args.min_active_decision_ratio,
            base.min_active_decision_ratio,
        ),
        min_trade_count=_coalesce(args.min_trade_count, base.min_trade_count),
        min_decision_change_count=_coalesce(
            args.min_decision_change_count,
            base.min_decision_change_count,
        ),
        min_positive_changed_decision_count=_coalesce(
            args.min_positive_changed_decision_count,
            base.min_positive_changed_decision_count,
        ),
        min_changed_return_delta=_coalesce(
            args.min_changed_return_delta,
            base.min_changed_return_delta,
        ),
    )


def _coalesce(value, fallback):
    return fallback if value is None else value


if __name__ == "__main__":
    sys.exit(main())
