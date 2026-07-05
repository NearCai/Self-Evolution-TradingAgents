"""Repair and summarize the A-share point-in-time baseline results.

The raw batch runner writes one row per ticker/date. If an agent run succeeded
but the post-hoc yfinance evaluation failed, this script can recover the row
from saved final_state files and local OHLCV cache, then write formal summary
tables for the research report.

Example:
    python scripts/summarize_baseline_ashare.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.default_config import DEFAULT_CONFIG

from scripts.run_baseline_ashare import (
    DEFAULT_UNIVERSE,
    RunSummary,
    action_direction,
    correctness,
    extract_trader_action,
    history_with_retry,
    realized_return,
    resolve_benchmark,
    write_csv,
)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def avg(values) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return mean(vals) if vals else None


def bool_rate(values) -> tuple[int, int, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0, 0, None
    hits = sum(1 for v in vals if bool(v))
    return hits, len(vals), hits / len(vals)


def read_rows(path: Path) -> list[RunSummary]:
    rows: list[RunSummary] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(RunSummary(**json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: list[RunSummary]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def expected_state_path(output_dir: Path, row: RunSummary) -> Path:
    analyst_key = row.analysts.replace(",", "+")
    return output_dir / row.ticker / row.analysis_date / analyst_key / "final_state.json"


def stock_return_only(ticker: str, trade_date: str, holding_days: int, actual_days: int | None = None):
    from datetime import datetime, timedelta

    start = datetime.strptime(trade_date, "%Y-%m-%d")
    end = start + timedelta(days=holding_days + 14)
    stock = history_with_retry(ticker, trade_date, end.strftime("%Y-%m-%d"))
    if len(stock) < 2:
        return None, None
    days = min(actual_days or holding_days, len(stock) - 1)
    value = float((stock["Close"].iloc[days] - stock["Close"].iloc[0]) / stock["Close"].iloc[0])
    return value, days


def benchmark_lookup(rows: list[RunSummary]) -> dict[tuple[str, str, int], tuple[float, int]]:
    lookup: dict[tuple[str, str, int], tuple[float, int]] = {}
    for row in rows:
        if row.benchmark and row.benchmark_return_5d is not None and row.actual_days_5d is not None:
            lookup[(row.benchmark, row.analysis_date, 5)] = (
                row.benchmark_return_5d,
                row.actual_days_5d,
            )
        if row.benchmark and row.benchmark_return_20d is not None and row.actual_days_20d is not None:
            lookup[(row.benchmark, row.analysis_date, 20)] = (
                row.benchmark_return_20d,
                row.actual_days_20d,
            )
    return lookup


def repair_from_state(row: RunSummary, output_dir: Path, bench_lookup: dict[tuple[str, str, int], tuple[float, int]]) -> RunSummary:
    state_path = expected_state_path(output_dir, row)
    if not state_path.exists():
        return row

    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = DEFAULT_CONFIG.copy()
    rating = parse_rating(state.get("final_trade_decision", ""), default="Hold")
    action = extract_trader_action(state, rating)
    direction = action_direction(action, rating)
    benchmark = resolve_benchmark(row.ticker, config)
    report_path = state_path.parent / "reports" / "complete_report.md"

    row.rating = rating
    row.trader_action = action
    row.signal_direction = direction
    row.benchmark = benchmark
    row.report_path = str(report_path) if report_path.exists() else row.report_path
    row.state_path = str(state_path)

    try:
        ret_5, bench_5, alpha_5, days_5 = realized_return(row.ticker, benchmark, row.analysis_date, 5)
        ret_20, bench_20, alpha_20, days_20 = realized_return(row.ticker, benchmark, row.analysis_date, 20)
        row.return_5d = ret_5
        row.benchmark_return_5d = bench_5
        row.alpha_5d = alpha_5
        row.actual_days_5d = days_5
        row.directional_correct_5d = correctness(direction, ret_5)
        row.alpha_correct_5d = correctness(direction, alpha_5)
        row.return_20d = ret_20
        row.benchmark_return_20d = bench_20
        row.alpha_20d = alpha_20
        row.actual_days_20d = days_20
        row.directional_correct_20d = correctness(direction, ret_20)
        row.alpha_correct_20d = correctness(direction, alpha_20)
        row.status = "ok_repaired"
        row.error = None
    except Exception as exc:
        repaired_any = False
        fallback_errors = []
        for holding_days in (5, 20):
            bench_key = (benchmark, row.analysis_date, holding_days)
            if bench_key not in bench_lookup:
                fallback_errors.append(f"missing benchmark fallback for {bench_key}")
                continue
            try:
                bench_ret, bench_days = bench_lookup[bench_key]
                stock_ret, stock_days = stock_return_only(
                    row.ticker,
                    row.analysis_date,
                    holding_days,
                    actual_days=bench_days,
                )
                if stock_ret is None:
                    fallback_errors.append(f"missing stock return for {holding_days}d")
                    continue
                alpha = stock_ret - bench_ret
                if holding_days == 5:
                    row.return_5d = stock_ret
                    row.benchmark_return_5d = bench_ret
                    row.alpha_5d = alpha
                    row.actual_days_5d = stock_days
                    row.directional_correct_5d = correctness(direction, stock_ret)
                    row.alpha_correct_5d = correctness(direction, alpha)
                else:
                    row.return_20d = stock_ret
                    row.benchmark_return_20d = bench_ret
                    row.alpha_20d = alpha
                    row.actual_days_20d = stock_days
                    row.directional_correct_20d = correctness(direction, stock_ret)
                    row.alpha_correct_20d = correctness(direction, alpha)
                repaired_any = True
            except Exception as fallback_exc:
                fallback_errors.append(f"{holding_days}d fallback failed: {fallback_exc!r}")
        if repaired_any and row.return_5d is not None and row.return_20d is not None:
            row.status = "ok_repaired"
            row.error = None
        else:
            row.status = "eval_error"
            details = "; ".join(fallback_errors)
            row.error = f"evaluation failed after state repair: {exc!r}; {details}"
    return row


def markdown_table(headers: list[str], data: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in data)
    return "\n".join(lines)


def summarize(rows: list[RunSummary]) -> str:
    successful = [r for r in rows if r.status in {"ok", "ok_repaired"}]
    active = [r for r in successful if r.signal_direction in (1, -1)]
    failures = [r for r in rows if r.status not in {"ok", "ok_repaired"}]

    lines: list[str] = []
    lines.append("# A-share Market-only Baseline Summary")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append(markdown_table(
        ["Item", "Value"],
        [
            ["Universe", "5 A-share stocks across consumer, manufacturing, new energy, banking, semiconductors"],
            ["Dates", "2025-03-31, 2025-04-30, 2025-05-30, 2025-06-30, 2025-09-30"],
            ["Analysts", "market"],
            ["LLM provider", DEFAULT_CONFIG["llm_provider"]],
            ["Quick model", DEFAULT_CONFIG["quick_think_llm"]],
            ["Deep model", DEFAULT_CONFIG["deep_think_llm"]],
            ["Memory", "Native TradingAgents memory/reflection"],
        ],
    ))
    lines.append("")

    status_counts = Counter(r.status for r in rows)
    lines.append("## Data Completeness")
    lines.append(markdown_table(
        ["Metric", "Value"],
        [
            ["Total planned runs", str(len(rows))],
            ["Successful evaluable runs", str(len(successful))],
            ["Failures / unavailable", str(len(failures))],
            ["Status counts", ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))],
        ],
    ))
    if failures:
        lines.append("")
        lines.append(markdown_table(
            ["Ticker", "Date", "Status", "Reason"],
            [[r.ticker, r.analysis_date, r.status, (r.error or "")[:120]] for r in failures],
        ))
    lines.append("")

    rating_counts = Counter(r.rating for r in successful)
    action_counts = Counter(r.trader_action for r in successful)
    lines.append("## Signal Distribution")
    lines.append(markdown_table(
        ["Type", "Counts"],
        [
            ["Rating", ", ".join(f"{k}: {v}" for k, v in sorted(rating_counts.items(), key=lambda x: str(x[0])))],
            ["Trader action", ", ".join(f"{k}: {v}" for k, v in sorted(action_counts.items(), key=lambda x: str(x[0])))],
            ["Active Buy/Sell signals", str(len(active))],
        ],
    ))
    lines.append("")

    acc_rows = []
    for key, label in [
        ("directional_correct_5d", "5d directional accuracy"),
        ("alpha_correct_5d", "5d alpha accuracy"),
        ("directional_correct_20d", "20d directional accuracy"),
        ("alpha_correct_20d", "20d alpha accuracy"),
    ]:
        hit, total, value = bool_rate(getattr(r, key) for r in successful)
        acc_rows.append([label, f"{hit}/{total}", pct(value)])

    signed_5 = avg(r.signal_direction * r.return_5d for r in active if r.return_5d is not None)
    signed_a5 = avg(r.signal_direction * r.alpha_5d for r in active if r.alpha_5d is not None)
    signed_20 = avg(r.signal_direction * r.return_20d for r in active if r.return_20d is not None)
    signed_a20 = avg(r.signal_direction * r.alpha_20d for r in active if r.alpha_20d is not None)
    lines.append("## Formal Accuracy")
    lines.append(markdown_table(["Metric", "Hit/Total", "Rate"], acc_rows))
    lines.append("")
    lines.append(markdown_table(
        ["Metric", "Value"],
        [
            ["Average 5d raw return", pct(avg(r.return_5d for r in successful))],
            ["Average 5d alpha", pct(avg(r.alpha_5d for r in successful))],
            ["Average 20d raw return", pct(avg(r.return_20d for r in successful))],
            ["Average 20d alpha", pct(avg(r.alpha_20d for r in successful))],
            ["Signed 5d return on Buy/Sell", pct(signed_5)],
            ["Signed 5d alpha on Buy/Sell", pct(signed_a5)],
            ["Signed 20d return on Buy/Sell", pct(signed_20)],
            ["Signed 20d alpha on Buy/Sell", pct(signed_a20)],
        ],
    ))
    lines.append("")

    by_ticker = defaultdict(list)
    for row in rows:
        by_ticker[row.ticker].append(row)
    ticker_rows = []
    for ticker in sorted(by_ticker):
        group = by_ticker[ticker]
        ok = [r for r in group if r.status in {"ok", "ok_repaired"}]
        d5 = bool_rate(r.directional_correct_5d for r in ok)
        a5 = bool_rate(r.alpha_correct_5d for r in ok)
        d20 = bool_rate(r.directional_correct_20d for r in ok)
        a20 = bool_rate(r.alpha_correct_20d for r in ok)
        ticker_rows.append([
            ticker,
            f"{len(ok)}/{len(group)}",
            pct(avg(r.return_5d for r in ok)),
            pct(avg(r.alpha_5d for r in ok)),
            f"{d5[0]}/{d5[1]}",
            f"{a5[0]}/{a5[1]}",
            pct(avg(r.return_20d for r in ok)),
            pct(avg(r.alpha_20d for r in ok)),
            f"{d20[0]}/{d20[1]}",
            f"{a20[0]}/{a20[1]}",
        ])
    lines.append("## By Ticker")
    lines.append(markdown_table(
        ["Ticker", "OK/Total", "Avg 5d Ret", "Avg 5d Alpha", "Dir 5d", "Alpha 5d", "Avg 20d Ret", "Avg 20d Alpha", "Dir 20d", "Alpha 20d"],
        ticker_rows,
    ))
    lines.append("")

    by_date = defaultdict(list)
    for row in successful:
        by_date[row.analysis_date].append(row)
    date_rows = []
    for date in sorted(by_date):
        group = by_date[date]
        date_rows.append([
            date,
            str(len(group)),
            pct(avg(r.return_5d for r in group)),
            pct(avg(r.alpha_5d for r in group)),
            pct(avg(r.return_20d for r in group)),
            pct(avg(r.alpha_20d for r in group)),
            ", ".join(f"{k}: {v}" for k, v in Counter(r.rating for r in group).items()),
        ])
    lines.append("## By Date")
    lines.append(markdown_table(
        ["Date", "N", "Avg 5d Ret", "Avg 5d Alpha", "Avg 20d Ret", "Avg 20d Alpha", "Ratings"],
        date_rows,
    ))
    lines.append("")

    lines.append("## Paper Comparison")
    lines.append(markdown_table(
        ["Dimension", "This A-share baseline", "TradingAgents paper"],
        [
            ["Market", "A-share: 600519.SS, 000333.SZ, 300750.SZ, 600036.SS, 688981.SS", "US stocks: AAPL, GOOGL, AMZN"],
            ["Period", "Five decision dates in 2025", "Daily simulation from 2024-01-01 to 2024-03-29"],
            ["Inputs", "market analyst only", "multi-modal: price, news, social sentiment, insider data, financial statements, technical indicators"],
            ["Evaluation", "5d/20d forward return and alpha after each point decision", "continuous back trading with executed daily buy/sell/hold signals"],
            ["Metrics", "directional accuracy, alpha accuracy, average return, signed return", "CR, ARR, Sharpe ratio, MDD"],
        ],
    ))
    lines.append("")
    lines.append(markdown_table(
        ["Paper ticker", "CR", "ARR", "Sharpe", "MDD"],
        [
            ["AAPL", "26.62%", "30.50%", "8.21", "0.91%"],
            ["GOOGL", "24.36%", "27.58%", "6.39", "1.69%"],
            ["AMZN", "23.21%", "24.90%", "5.60", "2.11%"],
        ],
    ))
    lines.append("")
    lines.append(markdown_table(
        ["Gap", "Interpretation"],
        [
            ["Metric gap", "Current results are signal-level and cannot be directly compared with paper CR/ARR/SR/MDD."],
            ["Data gap", "Current run uses only market data, while the paper uses a multi-modal analyst set."],
            ["Sample gap", "Current run has 25 A-share point decisions; the paper reports a roughly three-month daily trading simulation."],
            ["Performance signal", f"Current 20d active-signal directional accuracy is {acc_rows[2][2]}, and 20d active-signal signed alpha is {pct(signed_a20)}."],
            ["Next comparable experiment", "Run the continuous A-share backtest script and report CR, ARR, Sharpe, and MDD."],
        ],
    ))
    lines.append("")
    lines.append("Paper source: https://arxiv.org/html/2412.20138v6")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(DEFAULT_CONFIG["results_dir"]) / "baseline_ashare"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_jsonl = output_dir / "baseline_results.jsonl"
    repaired_jsonl = output_dir / "baseline_results_repaired.jsonl"
    repaired_csv = output_dir / "baseline_results_repaired.csv"
    summary_md = output_dir / "baseline_summary.md"

    rows = read_rows(raw_jsonl)
    lookup = benchmark_lookup(rows)
    repaired = [
        repair_from_state(row, output_dir, lookup) if row.status != "ok" else row
        for row in rows
    ]

    write_jsonl(repaired_jsonl, repaired)
    write_csv(repaired_csv, repaired)
    summary_md.write_text(summarize(repaired), encoding="utf-8")

    print("Wrote:", repaired_jsonl)
    print("Wrote:", repaired_csv)
    print("Wrote:", summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
