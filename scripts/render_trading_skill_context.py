"""Render a compact prompt context from candidate trading skills."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.evolution.skills import (  # noqa: E402
    load_candidate_skill_records,
    render_skill_context,
    select_candidate_skills,
)

DEFAULT_SKILLS_PATH = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
    / "evolution"
    / "skills"
    / "candidate_skills.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-jsonl",
        default=str(DEFAULT_SKILLS_PATH),
        help="Path to candidate_skills.jsonl.",
    )
    parser.add_argument("--rating", default=None, help="Optional proposed PM rating filter.")
    parser.add_argument("--execution-action", default=None, help="Optional proposed execution action filter.")
    parser.add_argument("--max-skills", type=int, default=3, help="Maximum skills to render.")
    parser.add_argument("--max-chars", type=int, default=1800, help="Maximum rendered context characters.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output file. Default: <skills-jsonl>/../skill_context.md.",
    )
    args = parser.parse_args()

    skills_path = Path(args.skills_jsonl)
    records = load_candidate_skill_records(skills_path)
    selected = select_candidate_skills(
        records,
        rating=args.rating,
        execution_action=args.execution_action,
        max_skills=args.max_skills,
    )
    context = render_skill_context(selected, max_chars=args.max_chars)

    output_file = Path(args.output_file) if args.output_file else skills_path.parent / "skill_context.md"
    output_file.write_text(context + ("\n" if context else ""), encoding="utf-8")

    print("Trading skill context renderer")
    print("Skills JSONL:", skills_path)
    print("Selected:", len(selected))
    print("Output file:", output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
