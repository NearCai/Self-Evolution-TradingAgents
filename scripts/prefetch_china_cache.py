"""Prefetch China A-share local caches for long backtest runs."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.china import (  # noqa: E402
    get_china_balance_sheet,
    get_china_cashflow,
    get_china_fundamentals,
    get_china_income_statement,
    load_china_ohlcv_range,
    resolve_china_benchmark,
)


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_benchmark(ticker: str) -> str:
    return resolve_china_benchmark(ticker) or "000001.SS"


def trading_dates(calendar_ticker: str, start_date: str, end_date: str) -> list[str]:
    history = load_china_ohlcv_range(calendar_ticker, start_date, end_date)
    if history.empty:
        return []
    dates = pd.to_datetime(history.index).strftime("%Y-%m-%d").tolist()
    return sorted(dict.fromkeys(dates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", required=True, help="Comma-separated A-share tickers.")
    parser.add_argument("--start-date", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--calendar-ticker",
        default="000001.SS",
        help="Ticker used to derive trading dates.",
    )
    parser.add_argument(
        "--include-statements",
        action="store_true",
        help="Also prefetch balance sheet, cashflow, and income statement text caches.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries per cache item.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between retries.")
    args = parser.parse_args()

    tickers = parse_csv_arg(args.tickers)
    dates = trading_dates(args.calendar_ticker, args.start_date, args.end_date)
    decision_dates = dates[:-1]
    price_end = (datetime.strptime(args.end_date, "%Y-%m-%d") + timedelta(days=5)).strftime(
        "%Y-%m-%d"
    )
    print("China cache prefetch")
    print("Tickers:", ", ".join(tickers))
    print("Dates:", f"{args.start_date} to {args.end_date}")
    print("Decision dates:", len(decision_dates))
    print("Include statements:", args.include_statements)

    failures: list[str] = []

    price_symbols = sorted(set(tickers + [resolve_benchmark(ticker) for ticker in tickers]))
    for symbol in price_symbols:
        ok = _retry(
            f"OHLCV {symbol}",
            lambda symbol=symbol: load_china_ohlcv_range(symbol, args.start_date, price_end),
            args.retries,
            args.sleep_seconds,
            failures,
        )
        print(f"[{'ok' if ok else 'fail'}] OHLCV {symbol}")

    total_items = len(tickers) * len(decision_dates)
    done = 0
    for date in decision_dates:
        for ticker in tickers:
            done += 1
            ok = _retry(
                f"fundamentals {ticker} {date}",
                lambda ticker=ticker, date=date: get_china_fundamentals(ticker, date),
                args.retries,
                args.sleep_seconds,
                failures,
            )
            print(f"[{'ok' if ok else 'fail'}] fundamentals {done}/{total_items} {ticker} {date}")
            if not args.include_statements:
                continue
            for label, fn in (
                ("balance", get_china_balance_sheet),
                ("cashflow", get_china_cashflow),
                ("income", get_china_income_statement),
            ):
                stmt_ok = _retry(
                    f"{label} {ticker} {date}",
                    lambda fn=fn, ticker=ticker, date=date: fn(ticker, "quarterly", date),
                    args.retries,
                    args.sleep_seconds,
                    failures,
                )
                print(f"[{'ok' if stmt_ok else 'fail'}] {label} {ticker} {date}")

    if failures:
        print()
        print("Prefetch completed with failures:")
        for item in failures:
            print("-", item)
        return 1

    print("Prefetch completed successfully.")
    return 0


def _retry(name: str, fn, retries: int, sleep_seconds: float, failures: list[str]) -> bool:
    attempts = max(1, retries + 1)
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - prefetch should continue through flaky vendors
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[retry] {name} attempt {attempt}/{attempts}: {last_error}")
            if attempt < attempts:
                time.sleep(sleep_seconds)
    failures.append(f"{name}: {last_error}")
    return False


if __name__ == "__main__":
    sys.exit(main())
