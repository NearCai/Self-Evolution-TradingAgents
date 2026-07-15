import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_walkforward_skill_split.py"
SPEC = importlib.util.spec_from_file_location("walkforward_skill_split", SCRIPT_PATH)
split = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = split
SPEC.loader.exec_module(split)


FIELDS = [
    "ticker",
    "name",
    "board",
    "sector",
    "analysis_date",
    "next_date",
    "analysts",
    "llm_provider",
    "quick_model",
    "deep_model",
    "rating",
    "trader_action",
    "decision_source",
    "execution_action",
    "signal_direction",
    "position_before",
    "position_after",
    "stock_return_next",
    "strategy_return_next",
    "benchmark_return_next",
    "status",
    "error",
]


def _write_result(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True)
    with (root / "continuous_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {
                "ticker": "600519.SS",
                "name": "Kweichow Moutai",
                "board": "Shanghai Main",
                "sector": "Consumer staples / liquor",
                "analysts": "market,fundamentals",
                "llm_provider": "deepseek",
                "quick_model": "deepseek-v4-flash",
                "deep_model": "deepseek-v4-flash",
                "rating": "Hold",
                "trader_action": "Hold",
                "decision_source": "pm-rating",
                "execution_action": "Hold",
                "signal_direction": "0",
                "position_before": "999.0",
                "position_after": "999.0",
                "strategy_return_next": "999.0",
                "benchmark_return_next": "0.0",
                "status": "ok",
                "error": "",
            }
            base.update(row)
            writer.writerow(base)
    return root


@pytest.mark.unit
def test_stitch_recomputes_continuous_long_cash_metrics(tmp_path):
    baseline = _write_result(
        tmp_path / "baseline",
        [
            {
                "analysis_date": "2026-04-01",
                "next_date": "2026-04-02",
                "execution_action": "Buy",
                "stock_return_next": "0.10",
            }
        ],
    )
    may = _write_result(
        tmp_path / "may",
        [
            {
                "analysis_date": "2026-05-29",
                "next_date": "2026-06-01",
                "execution_action": "Hold",
                "stock_return_next": "0.10",
            }
        ],
    )
    june = _write_result(
        tmp_path / "june",
        [
            {
                "analysis_date": "2026-06-01",
                "next_date": "2026-06-02",
                "execution_action": "Sell",
                "stock_return_next": "0.10",
            }
        ],
    )
    output = tmp_path / "final"

    manifest = split.stitch_walkforward_result(
        baseline_dir=baseline,
        may_dir=may,
        june_dir=june,
        output_dir=output,
        tickers=["600519.SS"],
    )

    assert manifest["rows"] == 3
    rows = list(csv.DictReader((output / "continuous_decisions.csv").open(encoding="utf-8")))
    assert [row["source_segment"] for row in rows] == [
        "apr_baseline_train",
        "may_full_skill_validation",
        "june_skill_test",
    ]
    assert [float(row["position_before"]) for row in rows] == [0.0, 1.0, 1.0]
    assert [float(row["position_after"]) for row in rows] == [1.0, 1.0, 0.0]
    assert [float(row["strategy_return_next"]) for row in rows] == pytest.approx([0.10, 0.10, 0.0])

    metrics = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8")))
    portfolio = next(row for row in metrics if row["scope"] == "portfolio_strategy")
    assert float(portfolio["cumulative_return"]) == pytest.approx(0.21)
