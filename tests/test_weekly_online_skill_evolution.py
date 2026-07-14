import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_weekly_online_skill_evolution.py"
SPEC = importlib.util.spec_from_file_location("weekly_online_skill_evolution", SCRIPT_PATH)
weekly = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = weekly
SPEC.loader.exec_module(weekly)


def _write_baseline(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = [
        ("600519.SS", "2026-04-29", "2026-04-30"),
        ("000333.SZ", "2026-04-29", "2026-04-30"),
        ("600519.SS", "2026-04-30", "2026-05-06"),
        ("000333.SZ", "2026-04-30", "2026-05-06"),
        ("600519.SS", "2026-05-06", "2026-05-07"),
        ("000333.SZ", "2026-05-06", "2026-05-07"),
        ("600519.SS", "2026-05-07", "2026-05-08"),
        ("000333.SZ", "2026-05-07", "2026-05-08"),
        ("600519.SS", "2026-05-08", "2026-05-11"),
        ("000333.SZ", "2026-05-08", "2026-05-11"),
        ("600519.SS", "2026-05-11", "2026-05-12"),
        ("000333.SZ", "2026-05-11", "2026-05-12"),
    ]
    with (root / "continuous_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "analysis_date",
                "next_date",
                "status",
                "execution_action",
                "position_before",
                "position_after",
                "strategy_return_next",
                "stock_return_next",
                "benchmark_return_next",
            ],
        )
        writer.writeheader()
        for ticker, analysis_date, next_date in rows:
            writer.writerow(
                {
                    "ticker": ticker,
                    "analysis_date": analysis_date,
                    "next_date": next_date,
                    "status": "ok",
                    "execution_action": "Hold",
                    "position_before": "0.0",
                    "position_after": "0.0",
                    "strategy_return_next": "0.0",
                    "stock_return_next": "0.01",
                    "benchmark_return_next": "0.0",
                }
            )
    return root


@pytest.mark.unit
def test_build_weekly_windows_from_baseline_decision_calendar(tmp_path):
    baseline = _write_baseline(tmp_path / "baseline")

    decision_dates = weekly.read_decision_dates(baseline)
    windows = weekly.build_weekly_windows(
        decision_dates,
        online_start_date="2026-05-01",
        end_date="2026-05-31",
    )

    assert len(windows) == 2
    assert windows[0].first_decision_date == "2026-05-06"
    assert windows[0].last_decision_date == "2026-05-08"
    assert windows[0].run_end_date == "2026-05-11"
    assert windows[1].first_decision_date == "2026-05-11"
    assert windows[1].run_end_date == "2026-05-12"


@pytest.mark.unit
def test_seed_output_from_baseline_writes_pre_online_jsonl(tmp_path):
    baseline = _write_baseline(tmp_path / "baseline")
    output = tmp_path / "online"

    seeded = weekly.seed_output_from_baseline(
        baseline_dir=baseline,
        output_dir=output,
        global_start_date="2026-04-01",
        online_start_date="2026-05-01",
    )

    records = [
        json.loads(line)
        for line in (output / "continuous_decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert seeded == 4
    assert {record["analysis_date"] for record in records} == {"2026-04-29", "2026-04-30"}
    assert all(record["status"] == "ok" for record in records)
    assert records[0]["stock_return_next"] == pytest.approx(0.01)


@pytest.mark.unit
def test_copy_initial_skill_library_is_resume_safe(tmp_path):
    initial = tmp_path / "candidate_skills.jsonl"
    active = tmp_path / "library" / "accepted_skills.jsonl"
    initial.write_text('{"skill_id":"s1"}\n', encoding="utf-8")

    assert weekly.copy_initial_skill_library(initial, active)
    assert active.read_text(encoding="utf-8") == initial.read_text(encoding="utf-8")

    active.write_text('{"skill_id":"existing"}\n', encoding="utf-8")
    assert not weekly.copy_initial_skill_library(initial, active)
    assert "existing" in active.read_text(encoding="utf-8")


@pytest.mark.unit
def test_load_manifest_records_for_resume(tmp_path):
    output = tmp_path / "online"
    manifest_dir = output / "_weekly_evolution"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "weekly_manifest.json").write_text(
        json.dumps(
            {
                "experiment": "weekly_online_skill_evolution",
                "records": [
                    {"week_key": "2026-W19", "status": "ok"},
                    {"week_key": "2026-W20", "status": "failed:backtest"},
                ],
            }
        ),
        encoding="utf-8",
    )

    records = weekly.load_manifest_records(output)

    assert [record["week_key"] for record in records] == ["2026-W19", "2026-W20"]
