"""Run and stitch a fixed train/validation/test skill-agent split.

Default split:
- April 2026: baseline Agent decisions used to build experiences and candidate skills.
- May 2026: skill-injected validation, including the May 29 decision whose
  next trading interval ends on June 1.
- June 2026: held-out skill-agent test using skills accepted by the May verifier.

The script writes a new final continuous result directory and does not overwrite
the earlier final_2026_q2_walkforward_best_skill_agent result.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_continuous_backtest_ashare import (  # noqa: E402
    DecisionRow,
    build_metrics,
    target_position,
    write_csv,
)

DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
)
DEFAULT_WORK_ROOT = PROJECT_ROOT / "results" / "walkforward_2026_q2_full_may_split"
DEFAULT_FINAL_DIR = PROJECT_ROOT / "results" / "final_2026_q2_walkforward_full_may_skill_agent"
DEFAULT_TICKERS = "600519.SS,000333.SZ,600036.SS"
DEFAULT_ANALYSTS = "market,fundamentals"
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"


def run_command(args: list[str], *, check: bool = True) -> int:
    printable = " ".join(args)
    print(f"[cmd] {printable}", flush=True)
    completed = subprocess.run(args, cwd=PROJECT_ROOT, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def _row_field_names() -> set[str]:
    return {field.name for field in fields(DecisionRow)}


def _clean_value(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def read_segment_rows(
    csv_path: Path,
    *,
    segment_name: str,
    start_date: str,
    end_date: str,
) -> list[tuple[DecisionRow, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing segment CSV: {csv_path}")

    row_fields = _row_field_names()
    selected: list[tuple[DecisionRow, str]] = []
    bad_rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            date = raw.get("analysis_date") or ""
            if not (start_date <= date <= end_date):
                continue
            if raw.get("status") != "ok":
                bad_rows.append(raw)
                continue
            kwargs = {
                key: _clean_value(value)
                for key, value in raw.items()
                if key in row_fields
            }
            selected.append((DecisionRow(**kwargs), segment_name))

    if bad_rows:
        examples = ", ".join(
            f"{row.get('ticker')} {row.get('analysis_date')} status={row.get('status')}"
            for row in bad_rows[:5]
        )
        raise RuntimeError(f"Segment {segment_name} contains non-ok rows: {examples}")
    if not selected:
        raise RuntimeError(f"Segment {segment_name} selected no rows from {csv_path}")
    return selected


def stitch_walkforward_result(
    *,
    baseline_dir: Path,
    may_dir: Path,
    june_dir: Path,
    output_dir: Path,
    tickers: list[str],
    transaction_cost_bps: float = 0.0,
    allow_short: bool = False,
) -> dict:
    segments = [
        {
            "name": "apr_baseline_train",
            "source_dir": baseline_dir,
            "start": "2026-04-01",
            "end": "2026-04-30",
        },
        {
            "name": "may_full_skill_validation",
            "source_dir": may_dir,
            "start": "2026-05-01",
            "end": "2026-05-29",
        },
        {
            "name": "june_skill_test",
            "source_dir": june_dir,
            "start": "2026-06-01",
            "end": "2026-06-29",
        },
    ]

    rows_with_segments: list[tuple[DecisionRow, str]] = []
    manifest_segments = []
    for segment in segments:
        csv_path = Path(segment["source_dir"]) / "continuous_decisions.csv"
        segment_rows = read_segment_rows(
            csv_path,
            segment_name=str(segment["name"]),
            start_date=str(segment["start"]),
            end_date=str(segment["end"]),
        )
        rows_with_segments.extend(segment_rows)
        manifest_segments.append(
            {
                "name": segment["name"],
                "source": str(csv_path),
                "start": segment["start"],
                "end": segment["end"],
                "rows": len(segment_rows),
                "decision_dates": len({row.analysis_date for row, _ in segment_rows}),
            }
        )

    requested_tickers = set(tickers)
    rows_with_segments = [
        (row, segment)
        for row, segment in rows_with_segments
        if row.ticker in requested_tickers
    ]
    rows_with_segments.sort(key=lambda item: (item[0].analysis_date, item[0].ticker))

    transaction_cost_rate = transaction_cost_bps / 10000.0
    positions = dict.fromkeys(tickers, 0.0)
    strategy_equities = dict.fromkeys(tickers, 1.0)
    buy_hold_equities = dict.fromkeys(tickers, 1.0)
    benchmark_equities = dict.fromkeys(tickers, 1.0)

    stitched_rows: list[DecisionRow] = []
    csv_rows: list[dict] = []
    jsonl_rows: list[dict] = []
    for row, segment_name in rows_with_segments:
        ticker = row.ticker
        stock_return = _to_float(row.stock_return_next, f"{ticker} {row.analysis_date} stock_return_next")
        benchmark_return = _to_float(
            row.benchmark_return_next,
            f"{ticker} {row.analysis_date} benchmark_return_next",
        )
        before = positions.get(ticker, 0.0)
        after = target_position(row.execution_action or "Hold", before, allow_short)
        turnover = abs(after - before)
        cost = turnover * transaction_cost_rate
        strategy_return = after * stock_return - cost

        positions[ticker] = after
        strategy_equities[ticker] *= 1.0 + strategy_return
        buy_hold_equities[ticker] *= 1.0 + stock_return
        benchmark_equities[ticker] *= 1.0 + benchmark_return

        row.position_before = before
        row.position_after = after
        row.stock_return_next = stock_return
        row.benchmark_return_next = benchmark_return
        row.strategy_return_next = strategy_return
        row.transaction_cost = cost
        row.equity_after = strategy_equities[ticker]
        row.buy_hold_equity_after = buy_hold_equities[ticker]
        row.benchmark_equity_after = benchmark_equities[ticker]
        row.status = "ok"
        row.error = None

        payload = asdict(row)
        payload["source_segment"] = segment_name
        csv_rows.append(payload)
        jsonl_rows.append(payload)
        stitched_rows.append(row)

    if not stitched_rows:
        raise RuntimeError("No stitched rows were produced.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "continuous_decisions.csv", csv_rows)
    with (output_dir / "continuous_decisions.jsonl").open("w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics, daily = build_metrics(stitched_rows)
    write_csv(output_dir / "daily_portfolio.csv", daily)
    write_csv(output_dir / "metrics.csv", metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "daily": daily}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "name": output_dir.name,
        "description": (
            "Strict train/validation/test walk-forward stitch: April baseline/training, "
            "May full validation through 2026-05-29, June held-out test. Continuous "
            "long/cash positions and equities are recomputed after stitching."
        ),
        "tickers": tickers,
        "decision_dates": len({row.analysis_date for row in stitched_rows}),
        "rows": len(stitched_rows),
        "period_start": stitched_rows[0].analysis_date,
        "period_end": stitched_rows[-1].next_date,
        "segments": manifest_segments,
        "execution_policy": "long/short" if allow_short else "long/cash",
        "decision_source": "pm-rating",
        "transaction_cost_bps": transaction_cost_bps,
        "skill_method": (
            "April candidate skills injected into full-May validation; June uses "
            "research-gate accepted skills from May validation."
        ),
    }
    (output_dir / "final_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme_text(manifest), encoding="utf-8")
    return manifest


def _to_float(value: object, label: str) -> float:
    if value in (None, ""):
        raise RuntimeError(f"Missing numeric value: {label}")
    return float(value)


def _readme_text(manifest: dict) -> str:
    lines = [
        f"# {manifest['name']}",
        "",
        "This directory contains the fixed train/validation/test walk-forward skill-agent result.",
        "",
        "Important: April is used for baseline decisions and skill generation; May is the",
        "validation month; June is the held-out test month. Positions and equities are",
        "recomputed continuously after stitching the segments.",
        "",
        "## Segments",
        "",
    ]
    for segment in manifest["segments"]:
        lines.append(
            f"- {segment['name']}: {segment['start']} to {segment['end']}, "
            f"rows={segment['rows']}, decision_dates={segment['decision_dates']}"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Rows: {manifest['rows']}",
            f"- Decision dates: {manifest['decision_dates']}",
            f"- Period: {manifest['period_start']} to {manifest['period_end']}",
            f"- Tickers: {', '.join(manifest['tickers'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_all(args: argparse.Namespace) -> None:
    work_root = Path(args.work_root)
    baseline_dir = Path(args.baseline_dir)
    final_dir = Path(args.final_dir)
    tickers = [item.strip() for item in args.tickers.split(",") if item.strip()]

    experience_dir = work_root / "train_2026_04_experiences"
    skill_dir = work_root / "train_2026_04_skills"
    may_dir = work_root / "val_2026_05_full_april_skills"
    june_dir = work_root / "test_2026_06_full_may_accepted_skills"
    may_verification_dir = may_dir / "skill_verification_research_gate"
    accepted_skills = may_verification_dir / "accepted_skills.jsonl"

    if not args.skip_train:
        run_command(
            [
                sys.executable,
                "scripts/build_trading_experiences.py",
                "--result-dir",
                str(baseline_dir),
                "--output-dir",
                str(experience_dir),
                "--start-date",
                "2026-04-01",
                "--end-date",
                "2026-04-30",
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/generate_trading_skills.py",
                "--experience-jsonl",
                str(experience_dir / "trading_experiences.jsonl"),
                "--output-dir",
                str(skill_dir),
            ]
        )

    if not args.skip_may:
        run_command(
            continuous_backtest_command(
                output_dir=may_dir,
                skills_jsonl=skill_dir / "candidate_skills.jsonl",
                start_date="2026-05-01",
                end_date="2026-06-01",
                args=args,
            )
        )
        run_command(
            [
                sys.executable,
                "scripts/evaluate_continuous_backtest_baselines.py",
                "--output-dir",
                str(may_dir),
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/verify_trading_skills.py",
                "--gate-preset",
                args.gate_preset,
                "--baseline-dir",
                str(baseline_dir),
                "--evolved-dir",
                str(may_dir),
                "--skills-jsonl",
                str(skill_dir / "candidate_skills.jsonl"),
                "--output-dir",
                str(may_verification_dir),
            ]
        )

    if not accepted_skills.exists():
        raise SystemExit(
            f"Accepted skills not found: {accepted_skills}. "
            "May validation likely failed; inspect the verifier output before running June."
        )

    if not args.skip_june:
        run_command(
            continuous_backtest_command(
                output_dir=june_dir,
                skills_jsonl=accepted_skills,
                start_date="2026-06-01",
                end_date="2026-06-30",
                args=args,
            )
        )
        run_command(
            [
                sys.executable,
                "scripts/evaluate_continuous_backtest_baselines.py",
                "--output-dir",
                str(june_dir),
            ]
        )

    manifest = stitch_walkforward_result(
        baseline_dir=baseline_dir,
        may_dir=may_dir,
        june_dir=june_dir,
        output_dir=final_dir,
        tickers=tickers,
        transaction_cost_bps=args.transaction_cost_bps,
        allow_short=args.allow_short,
    )
    print("Wrote:", final_dir / "continuous_decisions.csv")
    print("Wrote:", final_dir / "daily_portfolio.csv")
    print("Wrote:", final_dir / "metrics.csv")
    print("Wrote:", final_dir / "metrics.json")
    print("Wrote:", final_dir / "final_manifest.json")
    print("Final rows:", manifest["rows"])
    print("Final decision dates:", manifest["decision_dates"])

    if not args.skip_final_eval:
        run_command(
            [
                sys.executable,
                "scripts/evaluate_continuous_backtest_baselines.py",
                "--output-dir",
                str(final_dir),
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/verify_trading_skills.py",
                "--gate-preset",
                args.gate_preset,
                "--baseline-dir",
                str(baseline_dir),
                "--evolved-dir",
                str(final_dir),
                "--skills-jsonl",
                str(accepted_skills),
                "--output-dir",
                str(final_dir / "skill_verification_research_gate"),
            ],
            check=False,
        )


def continuous_backtest_command(
    *,
    output_dir: Path,
    skills_jsonl: Path,
    start_date: str,
    end_date: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_continuous_backtest_ashare.py",
        "--tickers",
        args.tickers,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--analysts",
        args.analysts,
        "--llm-provider",
        args.llm_provider,
        "--quick-model",
        args.quick_model,
        "--deep-model",
        args.deep_model,
        "--decision-source",
        args.decision_source,
        "--memory-mode",
        args.memory_mode,
        "--memory-holding-days",
        str(args.memory_holding_days),
        "--output-dir",
        str(output_dir),
        "--evolution-skills-jsonl",
        str(skills_jsonl),
        "--evolution-skill-max-skills",
        str(args.evolution_skill_max_skills),
        "--evolution-skill-max-chars",
        str(args.evolution_skill_max_chars),
        "--evolution-skill-types",
        args.evolution_skill_types,
        "--disable-prediction-markets",
        "--disable-us-social-sources",
    ]
    if args.evolution_opportunity_gate:
        command.append("--evolution-opportunity-gate")
    if args.evolution_position_risk_gate:
        command.append("--evolution-position-risk-gate")
    if args.allow_short:
        command.append("--allow-short")
    if args.transaction_cost_bps:
        command.extend(["--transaction-cost-bps", str(args.transaction_cost_bps)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--final-dir", default=str(DEFAULT_FINAL_DIR))
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--analysts", default=DEFAULT_ANALYSTS)
    parser.add_argument("--llm-provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--quick-model", default=DEFAULT_MODEL)
    parser.add_argument("--deep-model", default=DEFAULT_MODEL)
    parser.add_argument("--decision-source", default="pm-rating")
    parser.add_argument("--memory-mode", default="experiment")
    parser.add_argument("--memory-holding-days", type=int, default=5)
    parser.add_argument("--evolution-skill-max-skills", type=int, default=3)
    parser.add_argument("--evolution-skill-max-chars", type=int, default=1800)
    parser.add_argument("--evolution-skill-types", default="opportunity,promote")
    parser.add_argument("--gate-preset", choices=["default", "research"], default="research")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--evolution-opportunity-gate", action="store_true")
    parser.add_argument("--evolution-position-risk-gate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-may", action="store_true")
    parser.add_argument("--skip-june", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--stitch-only",
        action="store_true",
        help="Only stitch existing May/June result dirs and recompute final metrics.",
    )
    args = parser.parse_args()

    if args.stitch_only:
        tickers = [item.strip() for item in args.tickers.split(",") if item.strip()]
        stitch_walkforward_result(
            baseline_dir=Path(args.baseline_dir),
            may_dir=Path(args.work_root) / "val_2026_05_full_april_skills",
            june_dir=Path(args.work_root) / "test_2026_06_full_may_accepted_skills",
            output_dir=Path(args.final_dir),
            tickers=tickers,
            transaction_cost_bps=args.transaction_cost_bps,
            allow_short=args.allow_short,
        )
        return 0

    run_all(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
