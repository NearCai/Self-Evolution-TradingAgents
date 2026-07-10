"""Verify a skill-injected trading run against a matched baseline window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.evolution.verifier import (  # noqa: E402
    VerificationThresholds,
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
        default=0.0,
        help="Minimum evolved cumulative-return improvement over the matched baseline.",
    )
    parser.add_argument(
        "--max-drawdown-worsening",
        type=float,
        default=0.005,
        help="Allowed max-drawdown worsening, absolute return units.",
    )
    parser.add_argument(
        "--max-cash-drag-delta",
        type=int,
        default=1,
        help="Allowed increase in cash-drag intervals.",
    )
    parser.add_argument(
        "--max-turnover-delta",
        type=float,
        default=0.25,
        help="Allowed increase in average per-decision turnover.",
    )
    parser.add_argument(
        "--cash-up-threshold",
        type=float,
        default=0.005,
        help="Positive next-interval return threshold used to count cash-drag cases.",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    evolved_dir = Path(args.evolved_dir)
    skills_jsonl = Path(args.skills_jsonl) if args.skills_jsonl else None
    output_dir = Path(args.output_dir) if args.output_dir else evolved_dir / "skill_verification"
    thresholds = VerificationThresholds(
        min_return_delta=args.min_return_delta,
        max_drawdown_worsening=args.max_drawdown_worsening,
        max_cash_drag_delta=args.max_cash_drag_delta,
        max_turnover_delta=args.max_turnover_delta,
        cash_up_threshold=args.cash_up_threshold,
    )
    result = verify_skill_experiment(
        baseline_dir=baseline_dir,
        evolved_dir=evolved_dir,
        skills_jsonl=skills_jsonl,
        thresholds=thresholds,
    )
    files = write_verification_artifacts(result, output_dir, skills_jsonl=skills_jsonl)

    print("Trading skill verifier")
    print("Status:", "PASSED" if result.passed else "FAILED")
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
    for gate in result.gates:
        print(f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: {gate.message}")
    for path in files.values():
        print("Wrote:", path)
    return 0 if result.passed else 2


if __name__ == "__main__":
    sys.exit(main())
