"""Evaluate a continuous A-share backtest against paper-style rule baselines.

The TradingAgents paper reports CR/ARR/SR/MDD against Buy&Hold, MACD,
KDJ&RSI, ZMR, and SMA. This script reproduces that evaluation shape on a
completed ``run_continuous_backtest_ashare.py`` result directory.

Rules are executed in the same A-share realistic long/cash mode as the
continuous backtest by default: target position is either 1.0 (long) or 0.0
(cash). No future data is used; each decision uses indicators available at
``analysis_date`` and is applied to the next trading interval.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


TRADING_DAYS = 252.0
DEFAULT_OUTPUT_DIR = Path("results") / "continuous_ashare_2026_06_kimi_full"


@dataclass
class StrategyResult:
    method: str
    ticker: str
    analysis_date: str
    next_date: str
    action: str
    position_before: float
    position_after: float
    stock_return_next: float
    strategy_return_next: float
    equity_after: float


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


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
    method: str,
    ticker: str,
    returns: list[float],
    equities: list[float],
    positions: list[float] | None = None,
    turnovers: list[float] | None = None,
    actions: list[str] | None = None,
) -> dict:
    n = len(returns)
    if n == 0:
        return {
            "method": method,
            "ticker": ticker,
            "periods": 0,
            "CR": None,
            "ARR": None,
            "SR": None,
            "MDD": None,
            "win_rate": None,
            "avg_exposure": None,
            "avg_turnover": None,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }

    final_equity = equities[-1] if equities else math.prod(1.0 + r for r in returns)
    cumulative = final_equity - 1.0
    annualized = final_equity ** (TRADING_DAYS / n) - 1.0 if final_equity > 0 else None
    vol = stdev(returns) if n > 1 else 0.0
    sharpe = mean(returns) / vol * math.sqrt(TRADING_DAYS) if vol else None
    actions = actions or []
    positions = positions or []
    turnovers = turnovers or []
    return {
        "method": method,
        "ticker": ticker,
        "periods": n,
        "CR": cumulative,
        "ARR": annualized,
        "SR": sharpe,
        "MDD": max_drawdown([1.0] + equities),
        "win_rate": sum(1 for r in returns if r > 0) / n,
        "avg_exposure": mean(abs(p) for p in positions) if positions else None,
        "avg_turnover": mean(turnovers) if turnovers else None,
        "buy_count": sum(1 for a in actions if a == "Buy"),
        "sell_count": sum(1 for a in actions if a == "Sell"),
        "hold_count": sum(1 for a in actions if a == "Hold"),
    }


def to_float(value) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def load_decisions(path: Path) -> pd.DataFrame:
    decisions = pd.read_csv(path)
    decisions["analysis_date"] = pd.to_datetime(decisions["analysis_date"])
    decisions["next_date"] = pd.to_datetime(decisions["next_date"])
    decisions = decisions[decisions["status"] == "ok"].copy()
    decisions["stock_return_next"] = decisions["stock_return_next"].map(to_float)
    return decisions.sort_values(["analysis_date", "ticker"]).reset_index(drop=True)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values("Date").copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["sma_fast"] = close.rolling(5, min_periods=5).mean()
    out["sma_slow"] = close.rolling(20, min_periods=20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi"] = 100 - (100 / (1 + rs))

    low9 = low.rolling(9, min_periods=9).min()
    high9 = high.rolling(9, min_periods=9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, pd.NA) * 100
    out["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(com=2, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    roll_mean = close.rolling(20, min_periods=20).mean()
    roll_std = close.rolling(20, min_periods=20).std()
    out["zscore"] = (close - roll_mean) / roll_std.replace(0, pd.NA)
    return out


def load_price_cache(ticker: str, cache_dir: Path) -> pd.DataFrame:
    cache_name = ticker.replace(".SS", "_SH").replace(".SH", "_SH").replace(".SZ", "_SZ")
    path = cache_dir / f"{cache_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file for {ticker}: {path}")
    data = pd.read_csv(path, parse_dates=["Date"])
    return add_indicators(data)


def crossed_above(prev_a: float, prev_b: float, curr_a: float, curr_b: float) -> bool:
    return pd.notna(prev_a) and pd.notna(prev_b) and pd.notna(curr_a) and pd.notna(curr_b) and prev_a <= prev_b and curr_a > curr_b


def crossed_below(prev_a: float, prev_b: float, curr_a: float, curr_b: float) -> bool:
    return pd.notna(prev_a) and pd.notna(prev_b) and pd.notna(curr_a) and pd.notna(curr_b) and prev_a >= prev_b and curr_a < curr_b


def signal_for(method: str, history: pd.DataFrame, current_pos: float) -> str:
    if method == "Buy&Hold":
        return "Buy"
    if len(history) < 2:
        return "Hold"
    prev = history.iloc[-2]
    curr = history.iloc[-1]

    if method == "MACD":
        if crossed_above(prev.macd, prev.macd_signal, curr.macd, curr.macd_signal):
            return "Buy"
        if crossed_below(prev.macd, prev.macd_signal, curr.macd, curr.macd_signal):
            return "Sell"
        if pd.notna(curr.macd) and pd.notna(curr.macd_signal):
            return "Buy" if curr.macd > curr.macd_signal and current_pos <= 0 else "Hold"
        return "Hold"

    if method == "SMA":
        if crossed_above(prev.sma_fast, prev.sma_slow, curr.sma_fast, curr.sma_slow):
            return "Buy"
        if crossed_below(prev.sma_fast, prev.sma_slow, curr.sma_fast, curr.sma_slow):
            return "Sell"
        if pd.notna(curr.sma_fast) and pd.notna(curr.sma_slow):
            return "Buy" if curr.sma_fast > curr.sma_slow and current_pos <= 0 else "Hold"
        return "Hold"

    if method == "KDJ&RSI":
        kdj_buy = crossed_above(prev.kdj_j, prev.kdj_k, curr.kdj_j, curr.kdj_k)
        kdj_sell = crossed_below(prev.kdj_j, prev.kdj_k, curr.kdj_j, curr.kdj_k)
        rsi = curr.rsi
        if pd.notna(rsi) and (kdj_buy and rsi < 55 or rsi < 30):
            return "Buy"
        if pd.notna(rsi) and (kdj_sell and rsi > 45 or rsi > 70):
            return "Sell"
        return "Hold"

    if method == "ZMR":
        z = curr.zscore
        if pd.isna(z):
            return "Hold"
        if z < -1.0:
            return "Buy"
        if z > 1.0:
            return "Sell"
        return "Hold"

    raise ValueError(f"Unknown method: {method}")


def target_position(action: str, current: float, allow_short: bool) -> float:
    if action == "Buy":
        return 1.0
    if action == "Sell":
        return -1.0 if allow_short else 0.0
    return current


def run_method(
    method: str,
    decisions: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    allow_short: bool,
) -> list[StrategyResult]:
    results: list[StrategyResult] = []
    positions = defaultdict(float)
    equities = defaultdict(lambda: 1.0)

    for row in decisions.itertuples(index=False):
        ticker = row.ticker
        date = row.analysis_date
        history = price_cache[ticker][price_cache[ticker]["Date"] <= date]
        before = positions[ticker]
        action = signal_for(method, history, before)
        after = target_position(action, before, allow_short)
        strategy_ret = after * float(row.stock_return_next)
        equities[ticker] *= 1.0 + strategy_ret
        positions[ticker] = after
        results.append(
            StrategyResult(
                method=method,
                ticker=ticker,
                analysis_date=date.date().isoformat(),
                next_date=row.next_date.date().isoformat(),
                action=action,
                position_before=before,
                position_after=after,
                stock_return_next=float(row.stock_return_next),
                strategy_return_next=strategy_ret,
                equity_after=equities[ticker],
            )
        )
    return results


def agent_results(decisions: pd.DataFrame, allow_short: bool) -> list[StrategyResult]:
    results: list[StrategyResult] = []
    positions = defaultdict(float)
    equities = defaultdict(lambda: 1.0)
    for row in decisions.itertuples(index=False):
        ticker = row.ticker
        before = positions[ticker]
        action = getattr(row, "execution_action", None) or row.trader_action
        after = target_position(action, before, allow_short)
        strategy_ret = after * float(row.stock_return_next)
        equities[ticker] *= 1.0 + strategy_ret
        positions[ticker] = after
        results.append(
            StrategyResult(
                method="TradingAgents",
                ticker=ticker,
                analysis_date=row.analysis_date.date().isoformat(),
                next_date=row.next_date.date().isoformat(),
                action=action,
                position_before=before,
                position_after=after,
                stock_return_next=float(row.stock_return_next),
                strategy_return_next=strategy_ret,
                equity_after=equities[ticker],
            )
        )
    return results


def build_metrics(results: list[StrategyResult]) -> tuple[list[dict], list[dict]]:
    metrics: list[dict] = []
    daily: list[dict] = []
    by_method_ticker: dict[tuple[str, str], list[StrategyResult]] = defaultdict(list)
    by_method_day: dict[tuple[str, str, str], list[StrategyResult]] = defaultdict(list)
    for result in results:
        by_method_ticker[(result.method, result.ticker)].append(result)
        by_method_day[(result.method, result.analysis_date, result.next_date)].append(result)

    for (method, ticker), rows in sorted(by_method_ticker.items()):
        rows.sort(key=lambda r: r.analysis_date)
        metrics.append(
            summarize_returns(
                method,
                ticker,
                [r.strategy_return_next for r in rows],
                [r.equity_after for r in rows],
                [r.position_after for r in rows],
                [abs(r.position_after - r.position_before) for r in rows],
                [r.action for r in rows],
            )
        )

    portfolio_returns: dict[str, list[float]] = defaultdict(list)
    portfolio_equities: dict[str, list[float]] = defaultdict(list)
    portfolio_equity = defaultdict(lambda: 1.0)
    for (method, analysis_date, next_date), rows in sorted(by_method_day.items()):
        daily_ret = mean(r.strategy_return_next for r in rows)
        portfolio_equity[method] *= 1.0 + daily_ret
        portfolio_returns[method].append(daily_ret)
        portfolio_equities[method].append(portfolio_equity[method])
        daily.append(
            {
                "method": method,
                "analysis_date": analysis_date,
                "next_date": next_date,
                "n_tickers": len(rows),
                "return": daily_ret,
                "equity": portfolio_equity[method],
                "buy_count": sum(1 for r in rows if r.action == "Buy"),
                "sell_count": sum(1 for r in rows if r.action == "Sell"),
                "hold_count": sum(1 for r in rows if r.action == "Hold"),
            }
        )

    for method in sorted(portfolio_returns):
        metrics.append(
            summarize_returns(
                method,
                "PORTFOLIO",
                portfolio_returns[method],
                portfolio_equities[method],
            )
        )
    return metrics, daily


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, metrics: list[dict], execution_policy: str) -> None:
    portfolio = [row for row in metrics if row["ticker"] == "PORTFOLIO"]
    portfolio.sort(key=lambda r: (r["method"] != "TradingAgents", r["method"]))
    lines = [
        "# Continuous A-share Backtest: Paper-style Baseline Evaluation",
        "",
        f"Execution policy: `{execution_policy}`.",
        "",
        "Metrics follow the TradingAgents paper table shape: CR (cumulative return), "
        "ARR (annualized return, 252 trading days), SR (annualized Sharpe ratio), "
        "and MDD (maximum drawdown).",
        "",
        "## Portfolio Metrics",
        "",
        "| Method | Periods | CR | ARR | SR | MDD | Win Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in portfolio:
        sr = "" if row["SR"] is None else f"{row['SR']:.2f}"
        lines.append(
            f"| {row['method']} | {row['periods']} | {_pct(row['CR'])} | "
            f"{_pct(row['ARR'])} | {sr} | {_pct(row['MDD'])} | {_pct(row['win_rate'])} |"
        )
    lines += [
        "",
        "## Strategy Definitions",
        "",
        "- Buy&Hold: target position is always long.",
        "- MACD: 12/26 EMA MACD with 9-day signal; buy/sell on MACD-signal crossovers, otherwise keep position.",
        "- KDJ&RSI: KDJ(9,3,3) plus RSI(14); buys on oversold/momentum recovery and sells on overbought/momentum deterioration.",
        "- ZMR: 20-day zero-mean-reversion z-score; buy below -1 and sell above +1.",
        "- SMA: 5/20 simple moving-average crossover.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(Path.home() / ".tradingagents" / "cache" / "china_ohlcv"))
    parser.add_argument("--allow-short", action="store_true", help="Use paper-like long/short execution. Default is A-share long/cash.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    decisions_path = output_dir / "continuous_decisions.csv"
    decisions = load_decisions(decisions_path)
    tickers = sorted(decisions["ticker"].unique())
    price_cache = {ticker: load_price_cache(ticker, Path(args.cache_dir)) for ticker in tickers}

    all_results: list[StrategyResult] = agent_results(decisions, args.allow_short)
    for method in ("Buy&Hold", "MACD", "KDJ&RSI", "ZMR", "SMA"):
        all_results.extend(run_method(method, decisions, price_cache, args.allow_short))

    metrics, daily = build_metrics(all_results)
    eval_dir = output_dir / ("paper_style_eval_long_short" if args.allow_short else "paper_style_eval_long_cash")
    write_csv(eval_dir / "paper_style_metrics.csv", metrics)
    write_csv(eval_dir / "paper_style_daily.csv", daily)
    write_csv(eval_dir / "paper_style_decisions.csv", [result.__dict__ for result in all_results])
    write_markdown(eval_dir / "paper_style_summary.md", metrics, "long/short" if args.allow_short else "long/cash")

    portfolio = [row for row in metrics if row["ticker"] == "PORTFOLIO"]
    portfolio.sort(key=lambda r: (r["method"] != "TradingAgents", r["method"]))
    print("Wrote:", eval_dir / "paper_style_metrics.csv")
    print("Wrote:", eval_dir / "paper_style_daily.csv")
    print("Wrote:", eval_dir / "paper_style_summary.md")
    print()
    for row in portfolio:
        sr = "" if row["SR"] is None else f"{row['SR']:.2f}"
        print(
            f"{row['method']:14s} CR={_pct(row['CR']):>8s} "
            f"ARR={_pct(row['ARR']):>9s} SR={sr:>6s} MDD={_pct(row['MDD']):>8s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
