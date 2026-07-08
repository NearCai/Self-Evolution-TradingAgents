"""Run a continuous A-share TradingAgents backtest.

This is closer to the paper-style simulation than the point-in-time baseline:
for each trading day in a period, run the agent, execute its signal for the next
trading interval, and compute portfolio metrics.

The default memory mode uses one shared experiment memory file. This preserves
TradingAgents' memory/reflection behavior across the month while avoiding
contamination from unrelated manual runs. It also enables the graph's
look-ahead-safe memory mode so a reflection is injected only after its outcome
window is available by the simulated date.

Examples:
    python scripts/run_continuous_backtest_ashare.py --dry-run
    python scripts/run_continuous_backtest_ashare.py --tickers 600519.SS --end-date 2026-06-05
    python scripts/run_continuous_backtest_ashare.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev

import yfinance as yf

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.china import (
    is_a_share_symbol,
    is_china_index_symbol,
    load_china_ohlcv_range,
    resolve_china_benchmark,
)
from tradingagents.dataflows.symbol_utils import normalize_symbol
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
        "ticker": "600036.SS",
        "name": "China Merchants Bank",
        "board": "Shanghai Main",
        "sector": "Banking",
    },
]

DEFAULT_START_DATE = "2026-04-01"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_LLM_PROVIDER = "glm-cn"
DEFAULT_QUICK_MODEL = "glm-5-turbo"
DEFAULT_DEEP_MODEL = "glm-5"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

A_SHARE_DATA_VENDOR_CONFIG = {
    "core_stock_apis": "china",
    "technical_indicators": "china",
    "fundamental_data": "china",
    "news_data": "china",
}

ACTION_RE = re.compile(r"\*\*Action\*\*\s*:\s*([A-Za-z]+)", re.IGNORECASE)
PROPOSAL_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\**([A-Za-z]+)\**", re.IGNORECASE)


@dataclass
class DecisionRow:
    ticker: str
    name: str
    board: str
    sector: str
    analysis_date: str
    next_date: str
    analysts: str
    llm_provider: str
    quick_model: str
    deep_model: str
    rating: str | None = None
    trader_action: str | None = None
    decision_source: str | None = None
    execution_action: str | None = None
    signal_direction: int | None = None
    position_before: float | None = None
    position_after: float | None = None
    close: float | None = None
    next_close: float | None = None
    price_source: str | None = None
    stock_return_next: float | None = None
    strategy_return_next: float | None = None
    benchmark: str | None = None
    benchmark_close: float | None = None
    benchmark_next_close: float | None = None
    benchmark_price_source: str | None = None
    benchmark_return_next: float | None = None
    transaction_cost: float | None = None
    equity_after: float | None = None
    buy_hold_equity_after: float | None = None
    benchmark_equity_after: float | None = None
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


def default_output_dir(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> Path:
    start_month = start_date[:7].replace("-", "_")
    end_month = end_date[:7].replace("-", "_")
    if start_month == end_month:
        suffix = start_month
    else:
        suffix = f"{start_month}_to_{end_month}"
    return PROJECT_ROOT / "results" / f"continuous_ashare_{suffix}"


def resolve_benchmark(ticker: str, config: dict) -> str:
    explicit = config.get("benchmark_ticker")
    if explicit:
        return explicit
    china_benchmark = resolve_china_benchmark(ticker)
    if china_benchmark:
        return china_benchmark
    benchmark_map = config.get("benchmark_map", {})
    upper = ticker.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def history_with_retry(ticker: str, start: str, end: str, retries: int = 2):
    last = None
    if is_a_share_symbol(ticker) or is_china_index_symbol(ticker):
        for attempt in range(retries + 1):
            try:
                end_inclusive = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                data = load_china_ohlcv_range(ticker, start, end_inclusive)
                if len(data) > 0:
                    source = data.attrs.get("source", "China")
                    indexed = data.set_index("Date")
                    indexed.attrs["source"] = source
                    return indexed
                last = "empty China A-share history"
            except Exception as exc:
                last = str(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{ticker}: {last}")

    canonical = normalize_symbol(ticker)
    for attempt in range(retries + 1):
        try:
            data = yf.Ticker(canonical).history(start=start, end=end, auto_adjust=False)
            if len(data) > 0:
                data.attrs["source"] = f"Yahoo Finance ({canonical})"
                return data
            last = "empty history"
        except Exception as exc:
            last = str(exc)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{ticker}: {last}")


def history_source(data) -> str:
    return str(getattr(data, "attrs", {}).get("source") or "unknown")


def normalize_history_index(data) -> dict[str, float]:
    closes: dict[str, float] = {}
    for idx, row in data.iterrows():
        day = idx.date().isoformat()
        close = row.get("Close")
        if close is not None and not math.isnan(float(close)):
            closes[day] = float(close)
    return closes


def trading_dates(calendar_ticker: str, start_date: str, end_date: str) -> list[str]:
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    data = history_with_retry(calendar_ticker, start_date, end_exclusive)
    return [idx.date().isoformat() for idx in data.index]


def extract_trader_action(final_state: dict, rating: str) -> str:
    trader_text = final_state.get("trader_investment_plan") or ""
    final_text = final_state.get("final_trade_decision") or ""
    for text in (trader_text, final_text):
        match = PROPOSAL_RE.search(text) or ACTION_RE.search(text)
        if match:
            return match.group(1).capitalize()
    return rating_to_action(rating)


def rating_to_action(rating: str | None) -> str:
    rating_map = {
        "Buy": "Buy",
        "Overweight": "Buy",
        "Hold": "Hold",
        "Underweight": "Sell",
        "Sell": "Sell",
    }
    return rating_map.get(rating or "", "Hold")


def execution_action_from_source(trader_action: str, rating: str, decision_source: str) -> str:
    if decision_source == "pm-rating":
        return rating_to_action(rating)
    return trader_action or "Hold"


def action_direction(action: str, rating: str = "") -> int:
    action = (action or "").lower()
    rating = (rating or "").lower()
    if action == "buy":
        return 1
    if action == "sell":
        return -1
    if not action or action == "hold":
        if rating in {"buy", "overweight"}:
            return 1
        if rating in {"sell", "underweight"}:
            return -1
    return 0


def target_position(action: str, current: float, allow_short: bool) -> float:
    action = (action or "").lower()
    if action == "buy":
        return 1.0
    if action == "sell":
        return -1.0 if allow_short else 0.0
    return current


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_decisions(jsonl_path: Path) -> list[DecisionRow]:
    rows: list[DecisionRow] = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(DecisionRow(**json.loads(line)))
    return rows


def decision_key(row: DecisionRow) -> tuple[str, str, str]:
    return (row.ticker, row.analysis_date, row.analysts)


def latest_decision_rows(rows: list[DecisionRow]) -> list[DecisionRow]:
    latest: dict[tuple[str, str, str], DecisionRow] = {}
    for row in rows:
        latest[decision_key(row)] = row
    return sorted(latest.values(), key=lambda r: (r.analysis_date, r.ticker, r.analysts))


def build_backtest_config(
    output_dir: Path,
    memory_mode: str,
    memory_holding_days: int,
    benchmark_ticker: str | None = None,
    enable_prediction_markets: bool = True,
    enable_us_social_sources: bool = True,
    llm_provider: str | None = None,
    quick_model: str | None = None,
    deep_model: str | None = None,
) -> dict:
    base_config = copy.deepcopy(DEFAULT_CONFIG)
    base_config["results_dir"] = str(output_dir / "_graph_logs")
    base_config["memory_lookahead_safe"] = True
    base_config["memory_outcome_holding_days"] = memory_holding_days
    base_config["benchmark_ticker"] = benchmark_ticker
    base_config["enable_prediction_markets"] = enable_prediction_markets
    base_config["enable_us_social_sources"] = enable_us_social_sources
    if llm_provider:
        base_config["llm_provider"] = llm_provider
    if quick_model:
        base_config["quick_think_llm"] = quick_model
    if deep_model:
        base_config["deep_think_llm"] = deep_model

    data_vendors = dict(base_config.get("data_vendors", {}))
    data_vendors.update(A_SHARE_DATA_VENDOR_CONFIG)
    base_config["data_vendors"] = data_vendors

    if memory_mode == "experiment":
        base_config["memory_log_path"] = str(output_dir / "_memory" / "continuous_memory.md")
    elif memory_mode == "none":
        base_config["memory_log_path"] = None
    return base_config


def run_agent_decision(
    item: dict,
    date: str,
    next_date: str,
    analysts: list[str],
    output_dir: Path,
    config: dict,
    debug: bool,
    decision_source: str,
) -> tuple[DecisionRow, dict | None]:
    started = datetime.now()
    ticker = item["ticker"]
    analyst_key = ",".join(analysts)
    summary = DecisionRow(
        ticker=ticker,
        name=item.get("name", ""),
        board=item.get("board", ""),
        sector=item.get("sector", ""),
        analysis_date=date,
        next_date=next_date,
        analysts=analyst_key,
        llm_provider=config["llm_provider"],
        quick_model=config["quick_think_llm"],
        deep_model=config["deep_think_llm"],
        decision_source=decision_source,
        started_at=started.isoformat(timespec="seconds"),
    )

    safe_ticker = safe_ticker_component(ticker)
    run_dir = output_dir / safe_ticker / date / analyst_key.replace(",", "+")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        graph = TradingAgentsGraph(selected_analysts=tuple(analysts), debug=debug, config=config)
        final_state, decision = graph.propagate(ticker, date, asset_type="stock")
        rating = parse_rating(final_state.get("final_trade_decision", ""), default=decision or "Hold")
        trader_action = extract_trader_action(final_state, rating)
        execution_action = execution_action_from_source(trader_action, rating, decision_source)

        report_path = graph.save_reports(final_state, ticker, save_path=run_dir / "reports")
        state_path = run_dir / "final_state.json"
        state_path.write_text(json.dumps(final_state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        summary.rating = rating
        summary.trader_action = trader_action
        summary.decision_source = decision_source
        summary.execution_action = execution_action
        summary.signal_direction = action_direction(execution_action)
        summary.report_path = str(report_path)
        summary.state_path = str(state_path)
        final_state_out = final_state
    except BaseException as exc:  # noqa: BLE001 - batch runs must record provider/library exits
        if isinstance(exc, KeyboardInterrupt):
            raise
        summary.status = "error"
        summary.error = repr(exc)
        final_state_out = None

    finished = datetime.now()
    summary.finished_at = finished.isoformat(timespec="seconds")
    summary.elapsed_seconds = round((finished - started).total_seconds(), 2)
    return summary, final_state_out


def max_drawdown(equity: list[float]) -> float | None:
    if not equity:
        return None
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def summarize_returns(
    scope: str,
    returns: list[float],
    equities: list[float],
    exposures: list[float] | None = None,
    turnovers: list[float] | None = None,
    actions: list[str] | None = None,
) -> dict:
    n = len(returns)
    if n == 0:
        return {
            "scope": scope,
            "periods": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "daily_mean_return": None,
            "daily_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
            "avg_exposure": None,
            "avg_turnover": None,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }

    final_equity = equities[-1] if equities else math.prod(1.0 + r for r in returns)
    cumulative = final_equity - 1.0
    annualized = (final_equity ** (252.0 / n) - 1.0) if final_equity > 0 else None
    vol = stdev(returns) if n > 1 else 0.0
    sharpe = (mean(returns) / vol * math.sqrt(252.0)) if vol else None
    wins = [r for r in returns if r > 0]
    actions = actions or []

    return {
        "scope": scope,
        "periods": n,
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "daily_mean_return": mean(returns),
        "daily_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown([1.0] + equities),
        "win_rate": len(wins) / n,
        "avg_exposure": mean([abs(x) for x in exposures]) if exposures else None,
        "avg_turnover": mean(turnovers) if turnovers else None,
        "buy_count": sum(1 for a in actions if (a or "").lower() == "buy"),
        "sell_count": sum(1 for a in actions if (a or "").lower() == "sell"),
        "hold_count": sum(1 for a in actions if (a or "").lower() == "hold"),
    }


def build_metrics(rows: list[DecisionRow]) -> tuple[list[dict], list[dict]]:
    ok_rows = [
        row for row in rows
        if row.status == "ok"
        and row.strategy_return_next is not None
        and row.stock_return_next is not None
        and row.benchmark_return_next is not None
    ]
    ok_rows.sort(key=lambda r: (r.analysis_date, r.ticker))

    metrics: list[dict] = []
    daily: list[dict] = []

    by_ticker: dict[str, list[DecisionRow]] = {}
    by_day: dict[tuple[str, str], list[DecisionRow]] = {}
    for row in ok_rows:
        by_ticker.setdefault(row.ticker, []).append(row)
        by_day.setdefault((row.analysis_date, row.next_date), []).append(row)

    for ticker, ticker_rows in sorted(by_ticker.items()):
        metrics.append({
            **summarize_returns(
                f"strategy:{ticker}",
                [float(r.strategy_return_next) for r in ticker_rows],
                [float(r.equity_after) for r in ticker_rows if r.equity_after is not None],
                [float(r.position_after) for r in ticker_rows if r.position_after is not None],
                [
                    abs(float(r.position_after) - float(r.position_before))
                    for r in ticker_rows
                    if r.position_after is not None and r.position_before is not None
                ],
                [r.trader_action or "" for r in ticker_rows],
            ),
            "ticker": ticker,
            "variant": "strategy",
        })
        metrics.append({
            **summarize_returns(
                f"buy_hold:{ticker}",
                [float(r.stock_return_next) for r in ticker_rows],
                [float(r.buy_hold_equity_after) for r in ticker_rows if r.buy_hold_equity_after is not None],
            ),
            "ticker": ticker,
            "variant": "buy_hold",
        })
        metrics.append({
            **summarize_returns(
                f"benchmark:{ticker}",
                [float(r.benchmark_return_next) for r in ticker_rows],
                [float(r.benchmark_equity_after) for r in ticker_rows if r.benchmark_equity_after is not None],
            ),
            "ticker": ticker,
            "variant": "benchmark",
        })

    strategy_equity = 1.0
    buy_hold_equity = 1.0
    benchmark_equity = 1.0
    portfolio_strategy_returns: list[float] = []
    portfolio_buy_hold_returns: list[float] = []
    portfolio_benchmark_returns: list[float] = []
    strategy_equities: list[float] = []
    buy_hold_equities: list[float] = []
    benchmark_equities: list[float] = []

    for (date, next_date), day_rows in sorted(by_day.items()):
        strategy_ret = mean(float(r.strategy_return_next) for r in day_rows)
        buy_hold_ret = mean(float(r.stock_return_next) for r in day_rows)
        benchmark_ret = mean(float(r.benchmark_return_next) for r in day_rows)
        strategy_equity *= 1.0 + strategy_ret
        buy_hold_equity *= 1.0 + buy_hold_ret
        benchmark_equity *= 1.0 + benchmark_ret
        portfolio_strategy_returns.append(strategy_ret)
        portfolio_buy_hold_returns.append(buy_hold_ret)
        portfolio_benchmark_returns.append(benchmark_ret)
        strategy_equities.append(strategy_equity)
        buy_hold_equities.append(buy_hold_equity)
        benchmark_equities.append(benchmark_equity)
        daily.append({
            "analysis_date": date,
            "next_date": next_date,
            "n_tickers": len(day_rows),
            "strategy_return": strategy_ret,
            "buy_hold_return": buy_hold_ret,
            "benchmark_return": benchmark_ret,
            "strategy_equity": strategy_equity,
            "buy_hold_equity": buy_hold_equity,
            "benchmark_equity": benchmark_equity,
            "buy_count": sum(1 for r in day_rows if (r.trader_action or "").lower() == "buy"),
            "sell_count": sum(1 for r in day_rows if (r.trader_action or "").lower() == "sell"),
            "hold_count": sum(1 for r in day_rows if (r.trader_action or "").lower() == "hold"),
        })

    metrics.append({
        **summarize_returns("portfolio_strategy", portfolio_strategy_returns, strategy_equities),
        "ticker": "PORTFOLIO",
        "variant": "strategy",
    })
    metrics.append({
        **summarize_returns("portfolio_buy_hold", portfolio_buy_hold_returns, buy_hold_equities),
        "ticker": "PORTFOLIO",
        "variant": "buy_hold",
    })
    metrics.append({
        **summarize_returns("portfolio_benchmark", portfolio_benchmark_returns, benchmark_equities),
        "ticker": "PORTFOLIO",
        "variant": "benchmark",
    })
    return metrics, daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker override.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Backtest start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Backtest end date, YYYY-MM-DD.")
    parser.add_argument("--analysts", default="market", help="Comma-separated analysts: market,social,news,fundamentals.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to results/continuous_ashare_<start_YYYY_MM>_to_<end_YYYY_MM>.")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER,
                        help="LLM provider for this run. Default: glm-cn.")
    parser.add_argument("--quick-model", default=DEFAULT_QUICK_MODEL,
                        help="Quick-thinking model for this run. Default: glm-5-turbo.")
    parser.add_argument("--deep-model", default=DEFAULT_DEEP_MODEL,
                        help="Deep-thinking model for this run. Default: glm-5.")
    parser.add_argument("--calendar-ticker", default="000001.SS", help="Ticker used to derive the A-share trading calendar.")
    parser.add_argument("--memory-mode", choices=["experiment", "native", "none"], default="experiment",
                        help="experiment=one shared run memory; native=default project memory; none=disable memory log.")
    parser.add_argument("--memory-holding-days", type=int, default=5,
                        help="Outcome window used by TradingAgents reflection memory.")
    parser.add_argument("--benchmark-ticker", default=None,
                        help="Optional single benchmark override. Default uses A-share board-aware benchmarks.")
    parser.add_argument("--decision-source", choices=["pm-rating", "trader"], default="pm-rating",
                        help="Which agent output drives execution. Default executes Portfolio Manager rating.")
    parser.add_argument("--allow-short", action="store_true", help="Map Sell to -1. Default is long/cash for A-shares.")
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0,
                        help="One-way transaction cost in basis points applied to turnover.")
    parser.add_argument("--disable-prediction-markets", action="store_true",
                        help="Do not expose Polymarket/prediction-market tools to the news analyst.")
    parser.add_argument("--disable-us-social-sources", action="store_true",
                        help="Do not fetch StockTwits/Reddit in the sentiment analyst.")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Stop after this many new agent calls; useful for smoke/resume validation.")
    parser.add_argument("--force", action="store_true", help="Re-run completed ticker/date rows.")
    parser.add_argument("--debug", action="store_true", help="Stream graph messages; verbose and slower.")
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment grid without calling LLMs.")
    args = parser.parse_args()

    analysts = parse_csv_arg(args.analysts, ["market"])
    tickers = parse_csv_arg(args.tickers, [item["ticker"] for item in DEFAULT_UNIVERSE])
    universe = [item for item in DEFAULT_UNIVERSE if item["ticker"] in tickers]
    known = {item["ticker"] for item in universe}
    for ticker in tickers:
        if ticker not in known:
            universe.append({"ticker": ticker, "name": "", "board": "", "sector": ""})

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.start_date, args.end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "continuous_decisions.jsonl"
    decisions_csv_path = output_dir / "continuous_decisions.csv"
    daily_csv_path = output_dir / "daily_portfolio.csv"
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_json_path = output_dir / "metrics.json"

    dates = trading_dates(args.calendar_ticker, args.start_date, args.end_date)
    decision_dates = dates[:-1]
    if len(dates) < 2:
        raise RuntimeError(f"Need at least 2 trading dates between {args.start_date} and {args.end_date}.")

    base_config = build_backtest_config(
        output_dir=output_dir,
        memory_mode=args.memory_mode,
        memory_holding_days=args.memory_holding_days,
        benchmark_ticker=args.benchmark_ticker,
        enable_prediction_markets=not args.disable_prediction_markets,
        enable_us_social_sources=not args.disable_us_social_sources,
        llm_provider=args.llm_provider,
        quick_model=args.quick_model,
        deep_model=args.deep_model,
    )

    price_end = (datetime.strptime(args.end_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
    prices: dict[str, dict[str, float]] = {}
    price_sources: dict[str, str] = {}
    benchmarks: dict[str, str] = {}
    benchmark_prices: dict[str, dict[str, float]] = {}
    benchmark_price_sources: dict[str, str] = {}
    for item in universe:
        ticker = item["ticker"]
        benchmark = resolve_benchmark(ticker, base_config)
        benchmarks[ticker] = benchmark
        ticker_history = history_with_retry(ticker, args.start_date, price_end)
        prices[ticker] = normalize_history_index(ticker_history)
        price_sources[ticker] = history_source(ticker_history)
        if benchmark not in benchmark_prices:
            benchmark_history = history_with_retry(benchmark, args.start_date, price_end)
            benchmark_prices[benchmark] = normalize_history_index(benchmark_history)
            benchmark_price_sources[benchmark] = history_source(benchmark_history)

    print("Continuous A-share backtest")
    print("Provider:", base_config["llm_provider"])
    print("Quick model:", base_config["quick_think_llm"])
    print("Deep model:", base_config["deep_think_llm"])
    print("Analysts:", ",".join(analysts))
    print("Dates:", f"{args.start_date} to {args.end_date}")
    print("Trading dates:", len(dates), "| Decision dates:", len(decision_dates))
    print("Runs:", len(universe) * len(decision_dates))
    if args.max_runs is not None:
        print("Run cap:", args.max_runs)
    print("Output:", output_dir)
    print("Memory mode:", args.memory_mode)
    print("Memory log:", base_config.get("memory_log_path"))
    print("Memory look-ahead safe:", base_config["memory_lookahead_safe"])
    print("Data vendors:", base_config["data_vendors"])
    print("Prediction markets enabled:", base_config["enable_prediction_markets"])
    print("US social sources enabled:", base_config["enable_us_social_sources"])
    print("Benchmark policy:", args.benchmark_ticker or "A-share board-aware")
    print("Decision source:", args.decision_source)
    print("Execution policy:", "long/short" if args.allow_short else "long/cash")
    print("Transaction cost bps:", args.transaction_cost_bps)
    print()
    for item in universe:
        ticker = item["ticker"]
        benchmark = benchmarks[ticker]
        print(
            f"- {ticker:10s} | {item.get('board', ''):14s} | {item.get('sector', '')} "
            f"| benchmark={benchmark} | price={price_sources.get(ticker, 'unknown')}"
        )
    print("First decision dates:", ", ".join(decision_dates[:5]))
    print("Last decision dates:", ", ".join(decision_dates[-5:]))

    if args.dry_run:
        return 0

    existing_rows = latest_decision_rows(load_decisions(jsonl_path))
    completed = {
        decision_key(row): row
        for row in existing_rows
        if (
            row.status == "ok"
            and row.strategy_return_next is not None
            and row.decision_source == args.decision_source
        )
    }
    positions = {item["ticker"]: 0.0 for item in universe}
    equities = {item["ticker"]: 1.0 for item in universe}
    buy_hold_equities = {item["ticker"]: 1.0 for item in universe}
    benchmark_equities = {item["ticker"]: 1.0 for item in universe}
    transaction_cost_rate = args.transaction_cost_bps / 10000.0
    new_runs = 0
    stop_requested = False

    for date_index, date in enumerate(decision_dates):
        next_date = dates[date_index + 1]
        for item in universe:
            if args.max_runs is not None and new_runs >= args.max_runs:
                stop_requested = True
                break
            ticker = item["ticker"]
            analyst_key = ",".join(analysts)
            key = (ticker, date, analyst_key)
            if key in completed and not args.force:
                row = completed[key]
                positions[ticker] = float(row.position_after or 0.0)
                equities[ticker] = float(row.equity_after or equities[ticker])
                buy_hold_equities[ticker] = float(row.buy_hold_equity_after or buy_hold_equities[ticker])
                benchmark_equities[ticker] = float(row.benchmark_equity_after or benchmark_equities[ticker])
                print(f"[skip] {ticker} {date} already completed")
                continue

            print(f"[run] {ticker} {date} -> {next_date} analysts={analyst_key}")
            row, _ = run_agent_decision(
                item,
                date,
                next_date,
                analysts,
                output_dir,
                base_config,
                args.debug,
                args.decision_source,
            )
            new_runs += 1

            close = prices[ticker].get(date)
            next_close = prices[ticker].get(next_date)
            benchmark = benchmarks[ticker]
            benchmark_close = benchmark_prices[benchmark].get(date)
            benchmark_next_close = benchmark_prices[benchmark].get(next_date)

            row.benchmark = benchmark
            row.close = close
            row.next_close = next_close
            row.price_source = price_sources.get(ticker)
            row.benchmark_close = benchmark_close
            row.benchmark_next_close = benchmark_next_close
            row.benchmark_price_source = benchmark_price_sources.get(benchmark)
            row.position_before = positions[ticker]

            if row.status == "ok" and None not in (close, next_close, benchmark_close, benchmark_next_close):
                stock_ret = (float(next_close) - float(close)) / float(close)
                bench_ret = (float(benchmark_next_close) - float(benchmark_close)) / float(benchmark_close)
                new_position = target_position(row.execution_action or "Hold", positions[ticker], args.allow_short)
                turnover = abs(new_position - positions[ticker])
                cost = turnover * transaction_cost_rate
                strategy_ret = new_position * stock_ret - cost

                positions[ticker] = new_position
                equities[ticker] *= 1.0 + strategy_ret
                buy_hold_equities[ticker] *= 1.0 + stock_ret
                benchmark_equities[ticker] *= 1.0 + bench_ret

                row.position_after = new_position
                row.stock_return_next = stock_ret
                row.strategy_return_next = strategy_ret
                row.benchmark_return_next = bench_ret
                row.transaction_cost = cost
                row.equity_after = equities[ticker]
                row.buy_hold_equity_after = buy_hold_equities[ticker]
                row.benchmark_equity_after = benchmark_equities[ticker]
            elif row.status == "ok":
                row.status = "error"
                row.error = f"Missing close data for {ticker} or {benchmark} on {date}->{next_date}"
                row.position_after = positions[ticker]

            append_jsonl(jsonl_path, asdict(row))
            status = "ok" if row.status == "ok" else f"error: {row.error}"
            print(
                f"[done] {ticker} {date} {status} "
                f"action={row.execution_action} trader_action={row.trader_action} "
                f"rating={row.rating} elapsed={row.elapsed_seconds}s"
            )
        if stop_requested:
            print(f"[stop] max-runs reached after {new_runs} new agent call(s).")
            break

    all_rows = latest_decision_rows(load_decisions(jsonl_path))
    all_dicts = [asdict(row) for row in all_rows]
    write_csv(decisions_csv_path, all_dicts)
    metrics, daily = build_metrics(all_rows)
    write_csv(metrics_csv_path, metrics)
    write_csv(daily_csv_path, daily)
    metrics_json_path.write_text(json.dumps({"metrics": metrics, "daily": daily}, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("Wrote:", decisions_csv_path)
    print("Wrote:", daily_csv_path)
    print("Wrote:", metrics_csv_path)
    print("Wrote:", metrics_json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
