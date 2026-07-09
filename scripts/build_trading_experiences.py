"""Build structured trading experiences from a continuous backtest result dir."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.evolution.experience import (  # noqa: E402
    build_experiences,
    write_experience_artifacts,
)

DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default=str(DEFAULT_RESULT_DIR),
        help="Continuous backtest result directory containing continuous_decisions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <result-dir>/evolution/experiences.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Max characters kept from each report/decision text section.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include non-ok rows. Default keeps only successful agent decisions.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / "evolution" / "experiences"

    experiences = build_experiences(
        result_dir,
        max_text_chars=args.max_text_chars,
        require_ok=not args.include_errors,
    )
    manifest = write_experience_artifacts(experiences, output_dir)

    print("Trading experience builder")
    print("Result dir:", result_dir)
    print("Output dir:", output_dir)
    print("Experiences:", manifest["experience_count"])
    print("Labels:", json.dumps(manifest["label_counts"], ensure_ascii=False, sort_keys=True))
    print("Tickers:", json.dumps(manifest["ticker_counts"], ensure_ascii=False, sort_keys=True))
    print("Wrote:", manifest["files"]["jsonl"])
    print("Wrote:", manifest["files"]["summary_csv"])
    print("Wrote:", manifest["files"]["manifest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
