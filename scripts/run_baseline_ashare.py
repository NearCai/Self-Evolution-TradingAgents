"""Run a small A-share TradingAgents baseline batch and summarize outcomes.

This script wraps the repository's single-run TradingAgentsGraph API into a
reproducible baseline experiment:

- fixed ticker/date grid
- repository-configured memory/reflection by default, preserving stock behavior
- report tree saved for each run
- CSV/JSONL summaries with 5d and 20d realized returns

Example:
    python scripts/run_baseline_ashare.py --dry-run
    python scripts/run_baseline_ashare.py
    python scripts/run_baseline_ashare.py --analysts market,news,fundamentals
    python scripts/run_baseline_ashare.py --isolated-memory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


DEFAULT_UNIVERSE = [
    {
        "ticker": "600519.SS",
        "name": "Kweichow Moutai",
        "board": "Shanghai Main",
        "sector": "Consumer staples / liquor",
    },
    {
        "ticker": "000333.SZ",
        "name": "Midea Group",
        "board": "Shenzhen Main",
        "sector": "Home appliances / manufacturing",
    },
    {
        "ticker": "300750.SZ",
        "name": "CATL",
        "board": "ChiNext",
        "sector": "New energy / batteries",
    },
    {
        "ticker": "600036.SS",
        "name": "China Merchants Bank",
        "board": "Shanghai Main",
        "sector": "Banking",
    },
    {
        "ticker": "688981.SS",
        "name": "SMIC",
        "board": "STAR Market",
        "sector": "Semiconductors",
    },
]

DEFAULT_DATES = [
    "2025-03-31",
    "2025-04-30",
    "2025-05-30",
    "2025-06-30",
    "2025-09-30",
]

ACTION_RE = re.compile(r"\*\*Action\*\*\s*:\s*([A-Za-z]+)", re.IGNORECASE)
PROPOSAL_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\**([A-Za-z]+)\**", re.IGNORECASE)


@dataclass
class RunSummary:
    ticker: str
    name: str
    board: str
    sector: str
    analysis_date: str
    analysts: str
    llm_provider: str
    quick_model: str
    deep_model: str
    rating: str | None = None
    trader_action: str | None = None
    signal_direction: int | None = None
    benchmark: str | None = None
    return_5d: float | None = None
    benchmark_return_5d: float | None = None
    alpha_5d: float | None = None
    actual_days_5d: int | None = None
    directional_correct_5d: bool | None = None
    alpha_correct_5d: bool | None = None
    return_20d: float | None = None
    benchmark_return_20d: float | None = None
    alpha_20d: float | None = None
    actual_days_20d: int | None = None
    directional_correct_20d: bool | None = None
    alpha_correct_20d: bool | None = None
    report_path: str | None = None
    state_path: str | None = None
    status: str = "ok"
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float | None = None


def parse_csv_arg(raw: str | None, default: Iterable[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def default_output_dir() -> Path:
    return Path(DEFAULT_CONFIG["results_dir"]) / "baseline_ashare"


def resolve_benchmark(ticker: str, config: dict) -> str:
    explicit = config.get("benchmark_ticker")
    if explicit:
        return explicit
    benchmark_map = config.get("benchmark_map", {})
    upper = ticker.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def history_with_retry(ticker: str, start: str, end: str, retries: int = 2):
    last = None
    for attempt in range(retries + 1):
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if len(data) > 0:
                return data
            last = "empty history"
        except Exception as exc:  # yfinance raises several request-layer errors
            last = str(exc)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    cached = history_from_cache(ticker, start, end)
    if cached is not None and len(cached) > 0:
        return cached
    raise RuntimeError(f"{ticker}: {last}")


def history_from_cache(ticker: str, start: str, end: str):
    cache_dir = Path(DEFAULT_CONFIG["data_cache_dir"])
    matches = sorted(cache_dir.glob(f"{ticker}-YFin-data-*.csv"), reverse=True)
    if not matches:
        return None
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    for path in matches:
        try:
            data = pd.read_csv(path, parse_dates=["Date"])
            data = data[(data["Date"] >= start_dt) & (data["Date"] < end_dt)]
            if data.empty:
                continue
            data = data.set_index("Date")
            return data
        except Exception:
            continue
    return None


def realized_return(ticker: str, benchmark: str, trade_date: str, holding_days: int):
    start = datetime.strptime(trade_date, "%Y-%m-%d")
    end = start + timedelta(days=holding_days + 14)
    end_str = end.strftime("%Y-%m-%d")
    stock = history_with_retry(ticker, trade_date, end_str)
    bench = history_with_retry(benchmark, trade_date, end_str)
    if len(stock) < 2 or len(bench) < 2:
        return None, None, None, None
    actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
    stock_ret = float((stock["Close"].iloc[actual_days] - stock["Close"].iloc[0]) / stock["Close"].iloc[0])
    bench_ret = float((bench["Close"].iloc[actual_days] - bench["Close"].iloc[0]) / bench["Close"].iloc[0])
    return stock_ret, bench_ret, stock_ret - bench_ret, actual_days


def extract_trader_action(final_state: dict, rating: str) -> str:
    trader_text = final_state.get("trader_investment_plan") or ""
    final_text = final_state.get("final_trade_decision") or ""
    for text in (trader_text, final_text):
        match = PROPOSAL_RE.search(text) or ACTION_RE.search(text)
        if match:
            return match.group(1).capitalize()
    rating_map = {
        "Buy": "Buy",
        "Overweight": "Buy",
        "Hold": "Hold",
        "Underweight": "Sell",
        "Sell": "Sell",
    }
    return rating_map.get(rating, "Hold")


def action_direction(action: str, rating: str) -> int:
    action = (action or "").lower()
    rating = (rating or "").lower()
    if action == "buy" or rating in {"buy", "overweight"}:
        return 1
    if action == "sell" or rating in {"sell", "underweight"}:
        return -1
    return 0


def correctness(direction: int | None, value: float | None) -> bool | None:
    if direction is None or direction == 0 or value is None:
        return None
    return direction * value > 0


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[RunSummary]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_completed(jsonl_path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in {"ok", "eval_error"}:
                completed.add((row["ticker"], row["analysis_date"], row["analysts"]))
    return completed


def run_one(
    item: dict,
    date: str,
    analysts: list[str],
    output_dir: Path,
    isolated_memory: bool,
    debug: bool,
) -> RunSummary:
    ticker = item["ticker"]
    started = datetime.now()
    analyst_key = ",".join(analysts)
    summary = RunSummary(
        ticker=ticker,
        name=item.get("name", ""),
        board=item.get("board", ""),
        sector=item.get("sector", ""),
        analysis_date=date,
        analysts=analyst_key,
        llm_provider=DEFAULT_CONFIG["llm_provider"],
        quick_model=DEFAULT_CONFIG["quick_think_llm"],
        deep_model=DEFAULT_CONFIG["deep_think_llm"],
        started_at=started.isoformat(timespec="seconds"),
    )

    safe_ticker = safe_ticker_component(ticker)
    run_dir = output_dir / safe_ticker / date / analyst_key.replace(",", "+")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = DEFAULT_CONFIG.copy()
    config["results_dir"] = str(output_dir / "_graph_logs")
    if isolated_memory:
        config["memory_log_path"] = str(output_dir / "_memory" / f"{safe_ticker}_{date}_{analyst_key}.md")

    try:
        graph = TradingAgentsGraph(selected_analysts=tuple(analysts), debug=debug, config=config)
        final_state, decision = graph.propagate(ticker, date, asset_type="stock")
        rating = parse_rating(final_state.get("final_trade_decision", ""), default=decision or "Hold")
        action = extract_trader_action(final_state, rating)
        direction = action_direction(action, rating)
        benchmark = resolve_benchmark(ticker, config)

        report_path = graph.save_reports(final_state, ticker, save_path=run_dir / "reports")
        state_path = run_dir / "final_state.json"
        state_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        summary.rating = rating
        summary.trader_action = action
        summary.signal_direction = direction
        summary.benchmark = benchmark
        summary.report_path = str(report_path)
        summary.state_path = str(state_path)

        try:
            ret_5, bench_5, alpha_5, days_5 = realized_return(ticker, benchmark, date, 5)
            ret_20, bench_20, alpha_20, days_20 = realized_return(ticker, benchmark, date, 20)

            summary.return_5d = ret_5
            summary.benchmark_return_5d = bench_5
            summary.alpha_5d = alpha_5
            summary.actual_days_5d = days_5
            summary.directional_correct_5d = correctness(direction, ret_5)
            summary.alpha_correct_5d = correctness(direction, alpha_5)
            summary.return_20d = ret_20
            summary.benchmark_return_20d = bench_20
            summary.alpha_20d = alpha_20
            summary.actual_days_20d = days_20
            summary.directional_correct_20d = correctness(direction, ret_20)
            summary.alpha_correct_20d = correctness(direction, alpha_20)
        except Exception as exc:
            summary.status = "eval_error"
            summary.error = f"evaluation failed after successful agent run: {exc!r}"
    except Exception as exc:
        summary.status = "error"
        summary.error = repr(exc)

    finished = datetime.now()
    summary.finished_at = finished.isoformat(timespec="seconds")
    summary.elapsed_seconds = round((finished - started).total_seconds(), 2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker override.")
    parser.add_argument("--dates", default=None, help="Comma-separated date override, YYYY-MM-DD.")
    parser.add_argument("--analysts", default="market", help="Comma-separated analysts: market,social,news,fundamentals.")
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument(
        "--isolated-memory",
        action="store_true",
        help="Use a fresh memory log for each run. Default keeps the repository-configured memory/reflection behavior.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run completed ticker/date/analyst rows.")
    parser.add_argument("--debug", action="store_true", help="Stream graph messages; verbose and slower.")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment grid without calling LLMs.")
    args = parser.parse_args()

    analysts = parse_csv_arg(args.analysts, ["market"])
    dates = parse_csv_arg(args.dates, DEFAULT_DATES)
    tickers = parse_csv_arg(args.tickers, [item["ticker"] for item in DEFAULT_UNIVERSE])
    universe = [item for item in DEFAULT_UNIVERSE if item["ticker"] in tickers]
    known = {item["ticker"] for item in universe}
    for ticker in tickers:
        if ticker not in known:
            universe.append({"ticker": ticker, "name": "", "board": "", "sector": ""})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "baseline_results.jsonl"
    csv_path = output_dir / "baseline_results.csv"

    print("A-share baseline grid")
    print("Provider:", DEFAULT_CONFIG["llm_provider"])
    print("Quick model:", DEFAULT_CONFIG["quick_think_llm"])
    print("Deep model:", DEFAULT_CONFIG["deep_think_llm"])
    print("Analysts:", ",".join(analysts))
    print("Output:", output_dir)
    print("Memory mode:", "isolated per run" if args.isolated_memory else "native project memory/reflection")
    print("Memory log:", DEFAULT_CONFIG.get("memory_log_path") if not args.isolated_memory else output_dir / "_memory")
    print()
    for item in universe:
        print(f"- {item['ticker']:10s} | {item.get('board', ''):14s} | {item.get('sector', '')}")
    print("Dates:", ", ".join(dates))
    print("Runs:", len(universe) * len(dates))
    if args.dry_run:
        return 0

    completed = set() if args.force else load_completed(jsonl_path)
    rows: list[RunSummary] = []

    for item in universe:
        for date in dates:
            key = (item["ticker"], date, ",".join(analysts))
            if key in completed:
                print(f"[skip] {item['ticker']} {date} already completed")
                continue
            print(f"[run] {item['ticker']} {date} analysts={','.join(analysts)}")
            row = run_one(item, date, analysts, output_dir, args.isolated_memory, args.debug)
            rows.append(row)
            append_jsonl(jsonl_path, asdict(row))
            status = "ok" if row.status == "ok" else f"error: {row.error}"
            print(f"[done] {item['ticker']} {date} {status} elapsed={row.elapsed_seconds}s")

    all_rows: list[RunSummary] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_rows.append(RunSummary(**json.loads(line)))
    write_csv(csv_path, all_rows)
    print()
    print("Wrote:", csv_path)
    print("Wrote:", jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
