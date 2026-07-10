"""Generate candidate trading skills from structured trading experiences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.evolution.skills import (  # noqa: E402
    generate_candidate_skills,
    load_experience_records,
    write_skill_artifacts,
)

DEFAULT_EXPERIENCE_PATH = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
    / "evolution"
    / "experiences"
    / "trading_experiences.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experience-jsonl",
        default=str(DEFAULT_EXPERIENCE_PATH),
        help="Path to trading_experiences.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <experience-jsonl>/../skills.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Minimum evidence count for an action/rating group.",
    )
    parser.add_argument(
        "--promote-success-rate",
        type=float,
        default=0.55,
        help="Success-rate threshold for positive skills.",
    )
    parser.add_argument(
        "--warn-failure-rate",
        type=float,
        default=0.50,
        help="Failure-rate threshold for caution skills.",
    )
    parser.add_argument(
        "--missed-upside-return",
        type=float,
        default=0.005,
        help="Next-interval stock/benchmark return threshold for cash-drag opportunity skills.",
    )
    args = parser.parse_args()

    experience_path = Path(args.experience_jsonl)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experience_path.parent.parent / "skills"
    )

    experiences = load_experience_records(experience_path)
    skills = generate_candidate_skills(
        experiences,
        min_support=args.min_support,
        promote_success_rate=args.promote_success_rate,
        warn_failure_rate=args.warn_failure_rate,
        missed_upside_return=args.missed_upside_return,
    )
    manifest = write_skill_artifacts(skills, output_dir)

    print("Trading skill generator")
    print("Experience JSONL:", experience_path)
    print("Output dir:", output_dir)
    print("Skills:", manifest["skill_count"])
    print("Types:", json.dumps(manifest["type_counts"], ensure_ascii=False, sort_keys=True))
    print("Wrote:", manifest["files"]["jsonl"])
    print("Wrote:", manifest["files"]["markdown"])
    print("Wrote:", manifest["files"]["manifest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
