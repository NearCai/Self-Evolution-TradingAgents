"""Run weekly online skill evolution for the A-share continuous backtest.

This script orchestrates an online, no-lookahead skill loop on top of the
existing continuous backtest tools:

1. Seed the output with an earlier baseline/training period, usually April.
2. Run the next calendar week with the current accepted skill library.
3. Build experiences only from decisions that have already completed.
4. Generate a candidate skill library from cumulative past experiences.
5. Accept the new skill library only if the cumulative online run passes the
   verifier gate against the matched original baseline.
6. Use the accepted library in the following week.

The LLM-heavy work remains in ``run_continuous_backtest_ashare.py``. This file
only handles the week-by-week self-evolution control flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT
    / "results"
    / "continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm"
)
DEFAULT_INITIAL_SKILLS = (
    PROJECT_ROOT
    / "results"
    / "walkforward_2026_q2"
    / "train_2026_04_skills"
    / "candidate_skills.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "final_2026_q2_weekly_online_skill_agent"


@dataclass(frozen=True)
class DecisionDate:
    analysis_date: str
    next_date: str

    @property
    def parsed(self) -> date:
        return date.fromisoformat(self.analysis_date)


@dataclass(frozen=True)
class WeeklyWindow:
    week_index: int
    week_key: str
    first_decision_date: str
    last_decision_date: str
    run_end_date: str
    decision_dates: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--initial-skills-jsonl", default=str(DEFAULT_INITIAL_SKILLS))
    parser.add_argument("--tickers", default="600519.SS,000333.SZ,600036.SS")
    parser.add_argument("--global-start-date", default="2026-04-01")
    parser.add_argument("--online-start-date", default="2026-05-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--analysts", default="market,fundamentals")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--quick-model", default="deepseek-v4-flash")
    parser.add_argument("--deep-model", default="deepseek-v4-flash")
    parser.add_argument("--decision-source", default="pm-rating")
    parser.add_argument("--memory-mode", default="experiment")
    parser.add_argument("--memory-holding-days", default="5")
    parser.add_argument("--evolution-skill-max-skills", default="3")
    parser.add_argument("--evolution-skill-max-chars", default="1800")
    parser.add_argument("--evolution-skill-types", default="opportunity,promote")
    parser.add_argument("--gate-preset", default="research", choices=["default", "research"])
    parser.add_argument("--missed-upside-return", default="0.005")
    parser.add_argument("--min-support", default="5")
    parser.add_argument("--max-weeks", type=int, default=None)
    parser.add_argument("--disable-prediction-markets", action="store_true", default=True)
    parser.add_argument("--enable-prediction-markets", action="store_true")
    parser.add_argument("--disable-us-social-sources", action="store_true", default=True)
    parser.add_argument("--enable-us-social-sources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing weekly output directory. Default behavior also resumes if files exist.",
    )
    return parser.parse_args()


def read_decision_dates(baseline_dir: Path) -> list[DecisionDate]:
    path = baseline_dir / "continuous_decisions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline decisions: {path}")
    seen: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            analysis_date = row.get("analysis_date", "")
            next_date = row.get("next_date", "")
            if analysis_date and next_date:
                seen.setdefault(analysis_date, next_date)
    return [
        DecisionDate(analysis_date=analysis_date, next_date=next_date)
        for analysis_date, next_date in sorted(seen.items())
    ]


def build_weekly_windows(
    decision_dates: list[DecisionDate],
    *,
    online_start_date: str,
    end_date: str,
    max_weeks: int | None = None,
) -> list[WeeklyWindow]:
    start = date.fromisoformat(online_start_date)
    end = date.fromisoformat(end_date)
    filtered = [item for item in decision_dates if start <= item.parsed < end]
    grouped: dict[tuple[int, int], list[DecisionDate]] = {}
    for item in filtered:
        iso = item.parsed.isocalendar()
        grouped.setdefault((iso.year, iso.week), []).append(item)

    windows: list[WeeklyWindow] = []
    for idx, (week_key, items) in enumerate(sorted(grouped.items()), start=1):
        if max_weeks is not None and idx > max_weeks:
            break
        items = sorted(items, key=lambda item: item.analysis_date)
        windows.append(
            WeeklyWindow(
                week_index=idx,
                week_key=f"{week_key[0]}-W{week_key[1]:02d}",
                first_decision_date=items[0].analysis_date,
                last_decision_date=items[-1].analysis_date,
                run_end_date=items[-1].next_date,
                decision_dates=[item.analysis_date for item in items],
            )
        )
    return windows


def seed_output_from_baseline(
    *,
    baseline_dir: Path,
    output_dir: Path,
    global_start_date: str,
    online_start_date: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "continuous_decisions.jsonl"
    if jsonl_path.exists():
        return 0

    start = date.fromisoformat(global_start_date)
    online_start = date.fromisoformat(online_start_date)
    baseline_csv = baseline_dir / "continuous_decisions.csv"
    seeded = 0
    with baseline_csv.open("r", encoding="utf-8-sig", newline="") as src, jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for row in csv.DictReader(src):
            analysis_date = row.get("analysis_date")
            if not analysis_date:
                continue
            parsed = date.fromisoformat(analysis_date)
            if start <= parsed < online_start:
                dst.write(json.dumps(_json_clean(row), ensure_ascii=False) + "\n")
                seeded += 1
    return seeded


def copy_initial_skill_library(initial_skills: Path, active_skills: Path) -> bool:
    if active_skills.exists():
        return False
    active_skills.parent.mkdir(parents=True, exist_ok=True)
    if initial_skills.exists() and initial_skills.stat().st_size > 0:
        shutil.copyfile(initial_skills, active_skills)
    else:
        active_skills.write_text("", encoding="utf-8")
    return True


def run_command(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool = False) -> int:
    printable = " ".join(cmd)
    if dry_run:
        print("[dry-run]", printable)
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[cmd]", printable)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        log.write(f"[exit] {return_code}\n")
    return return_code


def command_run_backtest(args: argparse.Namespace, output_dir: Path, skills_path: Path, run_end_date: str) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_continuous_backtest_ashare.py",
        "--tickers",
        args.tickers,
        "--start-date",
        args.global_start_date,
        "--end-date",
        run_end_date,
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
    ]
    if skills_path.exists() and skills_path.stat().st_size > 0:
        cmd.extend(
            [
                "--evolution-skills-jsonl",
                str(skills_path),
                "--evolution-skill-max-skills",
                str(args.evolution_skill_max_skills),
                "--evolution-skill-max-chars",
                str(args.evolution_skill_max_chars),
                "--evolution-skill-types",
                args.evolution_skill_types,
            ]
        )
    if args.disable_prediction_markets and not args.enable_prediction_markets:
        cmd.append("--disable-prediction-markets")
    if args.disable_us_social_sources and not args.enable_us_social_sources:
        cmd.append("--disable-us-social-sources")
    return cmd


def command_build_experiences(args: argparse.Namespace, output_dir: Path, week_dir: Path, end_date: str) -> list[str]:
    return [
        sys.executable,
        "scripts/build_trading_experiences.py",
        "--result-dir",
        str(output_dir),
        "--output-dir",
        str(week_dir / "experiences"),
        "--start-date",
        args.global_start_date,
        "--end-date",
        end_date,
    ]


def command_generate_skills(args: argparse.Namespace, week_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/generate_trading_skills.py",
        "--experience-jsonl",
        str(week_dir / "experiences" / "trading_experiences.jsonl"),
        "--output-dir",
        str(week_dir / "candidate_skills"),
        "--min-support",
        str(args.min_support),
        "--missed-upside-return",
        str(args.missed_upside_return),
    ]


def command_verify(args: argparse.Namespace, output_dir: Path, week_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/verify_trading_skills.py",
        "--gate-preset",
        args.gate_preset,
        "--baseline-dir",
        str(Path(args.baseline_dir)),
        "--evolved-dir",
        str(output_dir),
        "--skills-jsonl",
        str(week_dir / "candidate_skills" / "candidate_skills.jsonl"),
        "--output-dir",
        str(week_dir / "verification"),
    ]


def command_evaluate(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_continuous_backtest_baselines.py",
        "--output-dir",
        str(output_dir),
    ]


def promote_accepted_skills(week_dir: Path, active_skills: Path) -> bool:
    accepted = week_dir / "verification" / "accepted_skills.jsonl"
    if not accepted.exists() or accepted.stat().st_size == 0:
        return False
    active_skills.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(accepted, active_skills)
    return True


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest = {
        "experiment": "weekly_online_skill_evolution",
        "records": records,
    }
    (output_dir / "_weekly_evolution" / "weekly_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _json_clean(row: dict[str, str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            cleaned[key] = None
            continue
        cleaned[key] = _coerce_scalar(value)
    return cleaned


def _coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    evolution_dir = output_dir / "_weekly_evolution"
    active_skills = evolution_dir / "skill_library" / "accepted_skills.jsonl"

    decision_dates = read_decision_dates(baseline_dir)
    windows = build_weekly_windows(
        decision_dates,
        online_start_date=args.online_start_date,
        end_date=args.end_date,
        max_weeks=args.max_weeks,
    )
    if not windows:
        raise ValueError("No weekly windows were found for the requested dates.")

    print("Weekly online skill evolution")
    print("Baseline:", baseline_dir)
    print("Output:", output_dir)
    print("Initial skills:", args.initial_skills_jsonl)
    print("Windows:", len(windows))
    for window in windows:
        print(
            f"- week {window.week_index:02d} {window.week_key}: "
            f"{window.first_decision_date} -> {window.last_decision_date}, "
            f"run end={window.run_end_date}, decisions={len(window.decision_dates)}"
        )

    if args.dry_run:
        print("Dry-run only; no files will be modified and no LLM calls will be made.")

    records: list[dict[str, Any]] = []
    if not args.dry_run:
        seeded = seed_output_from_baseline(
            baseline_dir=baseline_dir,
            output_dir=output_dir,
            global_start_date=args.global_start_date,
            online_start_date=args.online_start_date,
        )
        copied = copy_initial_skill_library(Path(args.initial_skills_jsonl), active_skills)
        print("Seeded baseline rows:", seeded)
        print("Initialized active skills:", copied)

    for window in windows:
        week_dir = evolution_dir / f"week_{window.week_index:02d}_{window.week_key}"
        log_path = week_dir / "weekly_online.log"
        record = asdict(window)
        record["active_skills_before"] = str(active_skills)
        skills_for_week = (
            Path(args.initial_skills_jsonl)
            if args.dry_run and not active_skills.exists()
            else active_skills
        )

        commands = [
            command_run_backtest(args, output_dir, skills_for_week, window.run_end_date),
            command_build_experiences(args, output_dir, week_dir, window.last_decision_date),
            command_generate_skills(args, week_dir),
            command_verify(args, output_dir, week_dir),
        ]
        if args.dry_run:
            for cmd in commands:
                run_command(cmd, cwd=PROJECT_ROOT, log_path=log_path, dry_run=True)
            record["status"] = "dry-run"
            records.append(record)
            continue

        failed = False
        for step_name, cmd in zip(
            ["backtest", "experience", "skill_generation", "verification"],
            commands,
            strict=True,
        ):
            code = run_command(cmd, cwd=PROJECT_ROOT, log_path=log_path)
            record[f"{step_name}_exit_code"] = code
            if code not in {0, 2} or (step_name != "verification" and code != 0):
                record["status"] = f"failed:{step_name}"
                failed = True
                break

        if failed:
            records.append(record)
            write_manifest(output_dir, records)
            return 1

        promoted = promote_accepted_skills(week_dir, active_skills)
        record["skill_library_updated"] = promoted
        record["active_skills_after"] = str(active_skills)
        record["status"] = "ok"
        records.append(record)
        write_manifest(output_dir, records)

    if not args.dry_run:
        eval_log = evolution_dir / "final_evaluation.log"
        code = run_command(command_evaluate(output_dir), cwd=PROJECT_ROOT, log_path=eval_log)
        if code != 0:
            return code
        write_manifest(output_dir, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
