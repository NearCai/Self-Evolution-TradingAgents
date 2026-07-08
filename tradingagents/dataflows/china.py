"""China A-share data vendor.

This module keeps the original TradingAgents dataflow shape while adding a
lightweight China vendor for A-share experiments. It uses optional local data
packages in a China-oriented fallback chain:

1. Tushare first when ``TUSHARE_TOKEN``/``TUSHARE_API_KEY`` is configured.
2. AKShare for Eastmoney OHLCV, financial statements, news, and comment scores.
3. BaoStock for stable historical OHLCV, valuation, and financial ratios.

All historical paths filter rows by the requested trade date before formatting
them for the agents.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError

logger = logging.getLogger(__name__)

_CACHE_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Volume", "Amount",
    "PctChange", "Turnover", "PE_TTM", "PB_MRQ", "PS_TTM", "PCF_NCF_TTM",
]


_SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
_BJ_PREFIXES = (
    "430", "831", "832", "833", "834", "835", "836", "837",
    "838", "839", "870", "871", "872", "873", "920",
)


@dataclass(frozen=True)
class AShareSymbol:
    code: str
    market: str

    @property
    def display(self) -> str:
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[self.market]
        return f"{self.code}.{suffix}"

    @property
    def yahoo(self) -> str:
        suffix = {"sh": "SS", "sz": "SZ", "bj": "BJ"}[self.market]
        return f"{self.code}.{suffix}"

    @property
    def tushare(self) -> str:
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[self.market]
        return f"{self.code}.{suffix}"

    @property
    def baostock(self) -> str:
        return f"{self.market}.{self.code}"


@dataclass(frozen=True)
class ChinaIndexSymbol:
    code: str
    market: str
    name: str

    @property
    def display(self) -> str:
        suffix = {"sh": "SS", "sz": "SZ"}[self.market]
        return f"{self.code}.{suffix}"

    @property
    def akshare(self) -> str:
        return f"{self.market}{self.code}"

    @property
    def baostock(self) -> str:
        return f"{self.market}.{self.code}"


_CHINA_INDEXES = {
    "000001.SS": ChinaIndexSymbol("000001", "sh", "SSE Composite Index"),
    "000001.SH": ChinaIndexSymbol("000001", "sh", "SSE Composite Index"),
    "SH.000001": ChinaIndexSymbol("000001", "sh", "SSE Composite Index"),
    "SH000001": ChinaIndexSymbol("000001", "sh", "SSE Composite Index"),
    "399001.SZ": ChinaIndexSymbol("399001", "sz", "SZSE Component Index"),
    "SZ.399001": ChinaIndexSymbol("399001", "sz", "SZSE Component Index"),
    "SZ399001": ChinaIndexSymbol("399001", "sz", "SZSE Component Index"),
    "399006.SZ": ChinaIndexSymbol("399006", "sz", "ChiNext Index"),
    "SZ.399006": ChinaIndexSymbol("399006", "sz", "ChiNext Index"),
    "SZ399006": ChinaIndexSymbol("399006", "sz", "ChiNext Index"),
}


_KNOWN_A_SHARE_IDENTITY = {
    "600519.SH": {
        "company_name": "Kweichow Moutai",
        "sector": "Consumer staples",
        "industry": "Liquor",
        "exchange": "Shanghai Main Board",
    },
    "000333.SZ": {
        "company_name": "Midea Group",
        "sector": "Consumer discretionary / manufacturing",
        "industry": "Home appliances",
        "exchange": "Shenzhen Main Board",
    },
    "300750.SZ": {
        "company_name": "CATL",
        "sector": "New energy",
        "industry": "Power batteries",
        "exchange": "ChiNext",
    },
    "600036.SH": {
        "company_name": "China Merchants Bank",
        "sector": "Financials",
        "industry": "Banking",
        "exchange": "Shanghai Main Board",
    },
    "688981.SH": {
        "company_name": "SMIC",
        "sector": "Information technology",
        "industry": "Semiconductors",
        "exchange": "STAR Market",
    },
}


def parse_china_index_symbol(symbol: str) -> ChinaIndexSymbol | None:
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    raw = symbol.strip().upper().lstrip("$").replace("_", ".")
    return _CHINA_INDEXES.get(raw)


def is_china_index_symbol(symbol: str) -> bool:
    return parse_china_index_symbol(symbol) is not None


def resolve_china_benchmark(symbol: str) -> str | None:
    """Return the A-share benchmark index for ``symbol``.

    The suffix-only default config maps every ``.SZ`` ticker to the Shenzhen
    Component Index. For board-aware A-share experiments we can be a little more
    precise without adding the heavy CN project stack:

    - Shanghai Main/STAR -> SSE Composite (000001.SS)
    - Shenzhen Main/SME -> SZSE Component (399001.SZ)
    - ChiNext (300/301) -> ChiNext Index (399006.SZ)
    """
    if is_china_index_symbol(symbol):
        parsed_index = parse_china_index_symbol(symbol)
        return parsed_index.display if parsed_index is not None else None

    parsed = parse_a_share_symbol(symbol)
    if parsed is None:
        return None
    if parsed.market == "sh":
        return "000001.SS"
    if parsed.market == "sz":
        if parsed.code.startswith(("300", "301")):
            return "399006.SZ"
        return "399001.SZ"
    if parsed.market == "bj":
        return "000001.SS"
    return None


def _infer_market(code: str) -> str | None:
    if code.startswith(_SH_PREFIXES):
        return "sh"
    if code.startswith(_SZ_PREFIXES):
        return "sz"
    if code.startswith(_BJ_PREFIXES):
        return "bj"
    return None


def parse_a_share_symbol(symbol: str) -> AShareSymbol | None:
    """Parse common A-share forms into code + exchange.

    Supported examples: ``600519.SS``, ``600519.SH``, ``000333.SZ``,
    ``sh.600519``, ``SZ000333``, and bare six-digit A-share codes.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    if parse_china_index_symbol(symbol) is not None:
        return None

    raw = symbol.strip().upper().lstrip("$")
    raw = raw.replace("_", ".")

    market: str | None = None
    code: str | None = None

    m = re.fullmatch(r"(SH|SZ|BJ)[.\-:]?(\d{6})", raw)
    if m:
        market = m.group(1).lower()
        code = m.group(2)
    else:
        m = re.fullmatch(r"(\d{6})(?:[.\-:](SS|SH|SZ|BJ))?", raw)
        if m:
            code = m.group(1)
            suffix = m.group(2)
            if suffix:
                market = "sh" if suffix in {"SS", "SH"} else suffix.lower()

    if code is None:
        return None
    if market is None:
        market = _infer_market(code)
    if market not in {"sh", "sz", "bj"}:
        return None
    return AShareSymbol(code=code, market=market)


def is_a_share_symbol(symbol: str) -> bool:
    return parse_a_share_symbol(symbol) is not None


def _require_a_share(symbol: str) -> AShareSymbol:
    parsed = parse_a_share_symbol(symbol)
    if parsed is None:
        raise NoMarketDataError(symbol, symbol, "not a supported China A-share symbol")
    return parsed


def resolve_china_identity(symbol: str) -> dict[str, str]:
    """Resolve A-share identity without yfinance.

    This is intentionally best-effort and fast. It avoids yfinance's ``.info``
    path, which can hang on A-share symbols behind some proxies.
    """
    index = parse_china_index_symbol(symbol)
    if index is not None:
        return {
            "company_name": index.name,
            "sector": "Broad market index",
            "industry": "Equity index",
            "exchange": "Shanghai Stock Exchange" if index.market == "sh" else "Shenzhen Stock Exchange",
            "quote_type": "INDEX",
        }

    parsed = parse_a_share_symbol(symbol)
    if parsed is None:
        return {}

    identity = {
        "exchange": "Shanghai Stock Exchange" if parsed.market == "sh" else "Shenzhen Stock Exchange",
        "quote_type": "EQUITY",
    }
    identity.update(_KNOWN_A_SHARE_IDENTITY.get(parsed.display, {}))
    with contextlib.suppress(Exception):
        basic = _fetch_baostock_basic(parsed)
        name = basic.get("Name")
        if name:
            identity.setdefault("company_name", name)
    return identity


def _date(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.Timestamp.today().normalize()
    parsed = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(parsed).normalize()


def _compact(value: pd.Timestamp | str) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def _ymd(value: pd.Timestamp | str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _safe_float(value):
    with contextlib.suppress(TypeError, ValueError):
        parsed = float(value)
        if pd.notna(parsed):
            return parsed
    return None


def _fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _truncate(text, limit: int = 500) -> str:
    if text is None or pd.isna(text):
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "AKShare is not installed. Install it with `pip install akshare`."
        ) from exc
    return ak


def _import_tushare():
    try:
        import tushare as ts
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "Tushare is not installed. Install it with `pip install tushare`."
        ) from exc
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY")
    if not token:
        raise VendorNotConfiguredError("TUSHARE_TOKEN is not configured.")
    ts.set_token(token)
    return ts


def _import_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "BaoStock is not installed. Install it with `pip install baostock`."
        ) from exc
    return bs


def _quiet_call(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def _normalize_ohlcv(
    raw: pd.DataFrame,
    source: str,
    symbol: AShareSymbol | ChinaIndexSymbol,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol.display, symbol.display, f"{source} returned no rows")

    df = raw.copy()
    rename = {
        "日期": "Date",
        "trade_date": "Date",
        "date": "Date",
        "开盘": "Open",
        "open": "Open",
        "最高": "High",
        "high": "High",
        "最低": "Low",
        "low": "Low",
        "收盘": "Close",
        "close": "Close",
        "成交量": "Volume",
        "volume": "Volume",
        "vol": "Volume",
        "成交额": "Amount",
        "amount": "Amount",
        "涨跌幅": "PctChange",
        "pctChg": "PctChange",
        "pct_chg": "PctChange",
        "换手率": "Turnover",
        "turn": "Turnover",
        "peTTM": "PE_TTM",
        "pbMRQ": "PB_MRQ",
        "psTTM": "PS_TTM",
        "pcfNcfTTM": "PCF_NCF_TTM",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise NoMarketDataError(
            symbol.display, symbol.display, f"{source} missing OHLCV columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"].astype(str), errors="coerce")
    for col in [
        "Open", "High", "Low", "Close", "Volume", "Amount", "PctChange",
        "Turnover", "PE_TTM", "PB_MRQ", "PS_TTM", "PCF_NCF_TTM",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    keep = [
        c for c in [
            "Date", "Open", "High", "Low", "Close", "Volume", "Amount",
            "PctChange", "Turnover", "PE_TTM", "PB_MRQ", "PS_TTM", "PCF_NCF_TTM",
        ] if c in df.columns
    ]
    df = df[keep].reset_index(drop=True)
    if df.empty:
        raise NoMarketDataError(symbol.display, symbol.display, f"{source} returned no usable rows")
    df.attrs["source"] = source
    return df


def _filter_ohlcv(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = _date(start_date)
    end = _date(end_date)
    out = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if out.empty:
        raise NoMarketDataError(
            "China A-share",
            "China A-share",
            f"no rows between {_ymd(start)} and {_ymd(end)}",
        )
    out.attrs["source"] = df.attrs.get("source", "China")
    return out


def _china_ohlcv_cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / "china_ohlcv"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _china_ohlcv_cache_key(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", symbol.upper()).strip("_")


def _china_ohlcv_cache_path(symbol: str) -> Path:
    return _china_ohlcv_cache_dir() / f"{_china_ohlcv_cache_key(symbol)}.csv"


def _normalize_cached_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if raw is None or raw.empty or "Date" not in raw.columns or "Close" not in raw.columns:
        return None

    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in _CACHE_COLUMNS:
        if col != "Date" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    keep = [col for col in _CACHE_COLUMNS if col in df.columns]
    if not keep or "Close" not in keep:
        return None
    df = df[keep].reset_index(drop=True)
    if df.empty:
        return None
    df.attrs["source"] = f"China OHLCV cache ({symbol})"
    return df


def _read_ohlcv_cache(symbol: str) -> pd.DataFrame | None:
    path = _china_ohlcv_cache_path(symbol)
    if not path.exists():
        return None
    try:
        return _normalize_cached_ohlcv(pd.read_csv(path), symbol)
    except Exception as exc:  # noqa: BLE001 - corrupt cache should not break live fetch
        logger.debug("Failed to read China OHLCV cache %s: %s", path, exc)
        return None


def _write_ohlcv_cache(symbol: str, fresh: pd.DataFrame) -> None:
    normalized = _normalize_cached_ohlcv(fresh, symbol)
    if normalized is None:
        return

    existing = _read_ohlcv_cache(symbol)
    frames = [frame for frame in (existing, normalized) if frame is not None and not frame.empty]
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    for col in _CACHE_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[_CACHE_COLUMNS]
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = (
        combined.dropna(subset=["Date", "Close"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")

    path = _china_ohlcv_cache_path(symbol)
    try:
        combined.to_csv(path, index=False)
    except Exception as exc:  # noqa: BLE001 - cache write is an optimization
        logger.debug("Failed to write China OHLCV cache %s: %s", path, exc)


def _cached_ohlcv_slice(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    require_full: bool,
    fallback: bool = False,
) -> pd.DataFrame | None:
    cached = _read_ohlcv_cache(symbol)
    if cached is None:
        return None
    try:
        out = _filter_ohlcv(cached, start_date, end_date)
    except NoMarketDataError:
        return None

    if require_full:
        dates = out["Date"].dropna()
        if dates.empty:
            return None
        start = _date(start_date)
        end = _date(end_date)
        # Weekend boundaries have no trading rows; weekdays should be covered
        # exactly so a daily backtest does not silently reuse yesterday's cache.
        boundary_slack = pd.Timedelta(days=3)
        start_ok = dates.min() <= (start if start.weekday() < 5 else start + boundary_slack)
        end_ok = dates.max() >= (end if end.weekday() < 5 else end - boundary_slack)
        if not start_ok or not end_ok:
            return None

    if fallback:
        out.attrs["source"] = f"China OHLCV cache ({symbol}; partial fallback after live vendors failed)"
    else:
        out.attrs["source"] = f"China OHLCV cache ({symbol})"
    return out


def _load_ohlcv_range_with_cache(
    request_symbol: str,
    cache_symbol: str,
    start_date: str,
    end_date: str,
    attempts,
) -> pd.DataFrame:
    cached = _cached_ohlcv_slice(cache_symbol, start_date, end_date, require_full=True)
    if cached is not None:
        return cached

    errors: list[str] = []
    for name, fetcher in attempts:
        try:
            data = fetcher()
            _write_ohlcv_cache(cache_symbol, data)
            return data
        except VendorNotConfiguredError as exc:
            errors.append(f"{name}: {exc}")
            continue
        except NoMarketDataError as exc:
            errors.append(f"{name}: {exc.detail or exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - next local vendor may work
            logger.debug("%s failed for %s: %s", name, cache_symbol, exc)
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

    fallback = _cached_ohlcv_slice(
        cache_symbol,
        start_date,
        end_date,
        require_full=False,
        fallback=True,
    )
    if fallback is not None:
        logger.warning(
            "Using partial China OHLCV cache for %s after live vendors failed: %s",
            cache_symbol,
            "; ".join(errors),
        )
        return fallback
    raise NoMarketDataError(request_symbol, cache_symbol, "; ".join(errors))


def _fetch_akshare_ohlcv(symbol: AShareSymbol, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_akshare()
    raw = ak.stock_zh_a_hist(
        symbol=symbol.code,
        period="daily",
        start_date=_compact(start_date),
        end_date=_compact(end_date),
        adjust="qfq",
    )
    df = _normalize_ohlcv(raw, "AKShare/Eastmoney", symbol)
    return _filter_ohlcv(df, start_date, end_date)


def _fetch_tushare_ohlcv(symbol: AShareSymbol, start_date: str, end_date: str) -> pd.DataFrame:
    ts = _import_tushare()
    if not hasattr(ts, "pro_bar"):
        raise VendorNotConfiguredError("Installed tushare does not expose pro_bar.")
    raw = ts.pro_bar(
        ts_code=symbol.tushare,
        adj="qfq",
        freq="D",
        start_date=_compact(start_date),
        end_date=_compact(end_date),
    )
    df = _normalize_ohlcv(raw, "Tushare", symbol)
    return _filter_ohlcv(df, start_date, end_date)


def _baostock_rows(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    return pd.DataFrame(rows, columns=rs.fields)


def _fetch_baostock_ohlcv(symbol: AShareSymbol, start_date: str, end_date: str) -> pd.DataFrame:
    if symbol.market == "bj":
        raise NoMarketDataError(symbol.display, symbol.display, "BaoStock does not cover BJ symbols")
    bs = _import_baostock()
    lg = _quiet_call(bs.login)
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_history_k_data_plus(
            symbol.baostock,
            (
                "date,code,open,high,low,close,preclose,volume,amount,turn,"
                "pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
            ),
            start_date=_ymd(start_date),
            end_date=_ymd(end_date),
            frequency="d",
            adjustflag="2",
        )
        raw = _baostock_rows(rs)
    finally:
        _quiet_call(bs.logout)
    df = _normalize_ohlcv(raw, "BaoStock", symbol)
    return _filter_ohlcv(df, start_date, end_date)


def _fetch_akshare_index_ohlcv(symbol: ChinaIndexSymbol, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _import_akshare()
    raw = ak.stock_zh_index_daily(symbol=symbol.akshare)
    df = _normalize_ohlcv(raw, "AKShare/China index", symbol)
    return _filter_ohlcv(df, start_date, end_date)


def _fetch_baostock_index_ohlcv(symbol: ChinaIndexSymbol, start_date: str, end_date: str) -> pd.DataFrame:
    bs = _import_baostock()
    lg = _quiet_call(bs.login)
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_history_k_data_plus(
            symbol.baostock,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=_ymd(start_date),
            end_date=_ymd(end_date),
            frequency="d",
            adjustflag="3",
        )
        raw = _baostock_rows(rs)
    finally:
        _quiet_call(bs.logout)
    df = _normalize_ohlcv(raw, "BaoStock/China index", symbol)
    return _filter_ohlcv(df, start_date, end_date)


def _load_china_index_ohlcv_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    parsed = parse_china_index_symbol(symbol)
    if parsed is None:
        raise NoMarketDataError(symbol, symbol, "not a supported China index symbol")
    return _load_ohlcv_range_with_cache(
        symbol,
        parsed.display,
        start_date,
        end_date,
        (
            ("AKShare index", lambda: _fetch_akshare_index_ohlcv(parsed, start_date, end_date)),
            ("BaoStock index", lambda: _fetch_baostock_index_ohlcv(parsed, start_date, end_date)),
        ),
    )


def china_ohlcv_vendor_chain() -> tuple[str, ...]:
    """Return the configured China OHLCV vendor chain.

    This mirrors the useful part of TradingAgents-CN's A-share design while
    staying lightweight. Tushare is preferred only when a token is present;
    otherwise we avoid an avoidable failed Tushare probe on every call.
    """
    has_tushare_token = bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_API_KEY"))
    if has_tushare_token:
        return ("Tushare", "AKShare", "BaoStock")
    return ("AKShare", "BaoStock")


def load_china_ohlcv_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    if is_china_index_symbol(symbol):
        return _load_china_index_ohlcv_range(symbol, start_date, end_date)
    parsed = _require_a_share(symbol)
    fetchers = {
        "Tushare": _fetch_tushare_ohlcv,
        "AKShare": _fetch_akshare_ohlcv,
        "BaoStock": _fetch_baostock_ohlcv,
    }
    return _load_ohlcv_range_with_cache(
        symbol,
        parsed.display,
        start_date,
        end_date,
        tuple(
            (name, lambda fetcher=fetchers[name]: fetcher(parsed, start_date, end_date))
            for name in china_ohlcv_vendor_chain()
        ),
    )


def load_china_ohlcv(symbol: str, curr_date: str, lookback_days: int = 365 * 5) -> pd.DataFrame:
    end = _date(curr_date)
    start = end - pd.Timedelta(days=lookback_days)
    return load_china_ohlcv_range(symbol, _ymd(start), _ymd(end))


def get_china_stock_data(
    symbol: Annotated[str, "China A-share ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    parsed = _require_a_share(symbol)
    data = load_china_ohlcv_range(symbol, start_date, end_date)
    out = data.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for col in ["Open", "High", "Low", "Close", "Amount", "PctChange", "Turnover"]:
        if col in out.columns:
            out[col] = out[col].round(4)

    source = data.attrs.get("source", "China")
    header = (
        f"# China A-share stock data for {parsed.display} (requested: {symbol}) "
        f"from {start_date} to {end_date}\n"
        f"# Source: {source}; adjustment: qfq/forward-adjusted when supported\n"
        f"# Total records: {len(out)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + out.to_csv(index=False)


INDICATOR_DESCRIPTIONS = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. Usage: Identify trend direction "
        "and serve as dynamic support/resistance."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend "
        "and identify golden/death cross setups."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. Usage: Capture quick shifts in "
        "momentum and potential entry points."
    ),
    "macd": "MACD: Computes momentum via differences of EMAs.",
    "macds": "MACD Signal: An EMA smoothing of the MACD line.",
    "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal.",
    "rsi": "RSI: Measures momentum to flag overbought/oversold conditions.",
    "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands.",
    "boll_ub": "Bollinger Upper Band: typically 2 standard deviations above the middle line.",
    "boll_lb": "Bollinger Lower Band: typically 2 standard deviations below the middle line.",
    "atr": "ATR: Averages true range to measure volatility.",
    "vwma": "VWMA: A moving average weighted by volume.",
    "mfi": "MFI: Money Flow Index uses price and volume to measure buying/selling pressure.",
}


def get_china_indicators(
    symbol: Annotated[str, "China A-share ticker symbol"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date in YYYY-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: "
            f"{list(INDICATOR_DESCRIPTIONS.keys())}"
        )

    curr = _date(curr_date)
    start = curr - pd.Timedelta(days=max(int(look_back_days) + 420, 460))
    data = load_china_ohlcv_range(symbol, _ymd(start), _ymd(curr))
    stock_df = wrap(data.copy())
    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.strftime("%Y-%m-%d")
    stock_df[indicator]

    values = {}
    for _, row in stock_df.iterrows():
        val = row.get(indicator)
        values[row["Date"]] = "N/A" if pd.isna(val) else str(val)

    before = curr - relativedelta(days=int(look_back_days))
    lines = []
    current = curr
    while current >= before:
        key = current.strftime("%Y-%m-%d")
        value = values.get(key, "N/A: Not a trading day (weekend or holiday)")
        lines.append(f"{key}: {value}")
        current -= relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {_ymd(curr)}:\n\n"
        + "\n".join(lines)
        + "\n\n"
        + INDICATOR_DESCRIPTIONS[indicator]
    )


def _fetch_baostock_basic(symbol: AShareSymbol) -> dict[str, str]:
    if symbol.market == "bj":
        return {}
    bs = _import_baostock()
    lg = _quiet_call(bs.login)
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        rs = bs.query_stock_basic(code=symbol.baostock)
        df = _baostock_rows(rs)
    finally:
        _quiet_call(bs.logout)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {
        "Name": row.get("code_name"),
        "BaoStock Code": row.get("code"),
        "IPO Date": row.get("ipoDate"),
        "Delist Date": row.get("outDate"),
        "Type": row.get("type"),
        "Status": row.get("status"),
    }


def _fetch_akshare_static_info(symbol: AShareSymbol) -> dict[str, str]:
    ak = _import_akshare()
    raw = ak.stock_individual_info_em(symbol=symbol.code)
    if raw is None or raw.empty or raw.shape[1] < 2:
        return {}
    info: dict[str, str] = {}
    for _, row in raw.iterrows():
        key = str(row.iloc[0]).strip()
        value = row.iloc[1]
        if pd.isna(value):
            continue
        mapped = {
            "股票简称": "Name",
            "行业": "Industry",
            "上市时间": "Listing Date",
            "总股本": "Total Shares",
            "流通股": "Float Shares",
        }.get(key)
        if mapped:
            info[mapped] = str(value)
    return info


def _latest_baostock_valuation(symbol: AShareSymbol, curr_date: str | None) -> dict[str, str]:
    end = _date(curr_date)
    start = end - pd.Timedelta(days=30)
    data = load_china_ohlcv_range(symbol.display, _ymd(start), _ymd(end))
    row = data.iloc[-1]
    return {
        "Valuation Date": _ymd(row["Date"]),
        "Close": _fmt(row.get("Close")),
        "PE TTM": _fmt(row.get("PE_TTM")),
        "PB MRQ": _fmt(row.get("PB_MRQ")),
        "PS TTM": _fmt(row.get("PS_TTM")),
        "PCF NCF TTM": _fmt(row.get("PCF_NCF_TTM")),
        "Turnover": _fmt(row.get("Turnover"), 4),
    }


def _fetch_tushare_daily_basic(symbol: AShareSymbol, curr_date: str | None) -> dict[str, str]:
    ts = _import_tushare()
    api = ts.pro_api()
    end = _date(curr_date)
    start = end - pd.Timedelta(days=30)
    raw = api.daily_basic(
        ts_code=symbol.tushare,
        start_date=_compact(start),
        end_date=_compact(end),
        fields="ts_code,trade_date,close,turnover_rate,pe,pe_ttm,pb,ps_ttm,total_mv,circ_mv",
    )
    if raw is None or raw.empty:
        return {}
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"].astype(str), errors="coerce")
    raw = raw.dropna(subset=["trade_date"]).sort_values("trade_date")
    if raw.empty:
        return {}
    row = raw.iloc[-1]
    return {
        "Tushare Trade Date": _ymd(row["trade_date"]),
        "PE": _fmt(row.get("pe")),
        "PE TTM": _fmt(row.get("pe_ttm")),
        "PB": _fmt(row.get("pb")),
        "PS TTM": _fmt(row.get("ps_ttm")),
        "Total Market Value (10k CNY)": _fmt(row.get("total_mv")),
        "Float Market Value (10k CNY)": _fmt(row.get("circ_mv")),
    }


def _quarter_ends_for(curr_date: str | None, years_back: int = 5):
    curr = _date(curr_date)
    for year in range(curr.year, curr.year - years_back - 1, -1):
        for quarter, month_day in ((4, "12-31"), (3, "09-30"), (2, "06-30"), (1, "03-31")):
            stat = pd.Timestamp(f"{year}-{month_day}")
            if stat <= curr:
                yield year, quarter, stat


def _fetch_baostock_financial_rows(
    symbol: AShareSymbol,
    kind: str,
    curr_date: str | None,
    freq: str = "quarterly",
    limit: int = 8,
) -> pd.DataFrame:
    if symbol.market == "bj":
        raise NoMarketDataError(symbol.display, symbol.display, "BaoStock does not cover BJ symbols")
    bs = _import_baostock()
    query = {
        "profit": bs.query_profit_data,
        "balance": bs.query_balance_data,
        "cashflow": bs.query_cash_flow_data,
    }[kind]
    curr = _date(curr_date)
    rows: list[pd.DataFrame] = []
    lg = _quiet_call(bs.login)
    if lg.error_code != "0":
        raise RuntimeError(lg.error_msg)
    try:
        for year, quarter, _ in _quarter_ends_for(curr_date):
            rs = query(code=symbol.baostock, year=year, quarter=quarter)
            df = _baostock_rows(rs)
            if not df.empty:
                rows.append(df)
    finally:
        _quiet_call(bs.logout)

    if not rows:
        raise NoMarketDataError(symbol.display, symbol.display, f"BaoStock {kind} returned no rows")
    out = pd.concat(rows, ignore_index=True)
    for col in ("pubDate", "statDate"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "pubDate" in out.columns:
        out = out[out["pubDate"] <= curr]
    if freq.lower() == "annual" and "statDate" in out.columns:
        out = out[out["statDate"].dt.strftime("%m-%d") == "12-31"]
    out = out.sort_values([c for c in ("pubDate", "statDate") if c in out.columns], ascending=False)
    out = out.head(limit).reset_index(drop=True)
    if out.empty:
        raise NoMarketDataError(
            symbol.display, symbol.display, f"no {kind} rows announced by {_ymd(curr)}"
        )
    return out


def _fetch_akshare_statement(
    symbol: AShareSymbol,
    statement_symbol: str,
    curr_date: str | None,
    freq: str = "quarterly",
    limit: int = 8,
) -> pd.DataFrame:
    ak = _import_akshare()
    raw = ak.stock_financial_report_sina(stock=symbol.code, symbol=statement_symbol)
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol.display, symbol.display, f"AKShare {statement_symbol} empty")
    df = raw.copy()
    curr = _date(curr_date)
    for col in ("公告日期", "报告日"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), errors="coerce")
    if "公告日期" in df.columns:
        df = df[df["公告日期"] <= curr]
    elif "报告日" in df.columns:
        df = df[df["报告日"] <= curr]
    if freq.lower() == "annual" and "报告日" in df.columns:
        df = df[df["报告日"].dt.strftime("%m-%d") == "12-31"]
    if df.empty:
        raise NoMarketDataError(
            symbol.display, symbol.display, f"no {statement_symbol} announced by {_ymd(curr)}"
        )
    sort_cols = [c for c in ("公告日期", "报告日") if c in df.columns]
    df = df.sort_values(sort_cols, ascending=False).head(limit)
    mostly_empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=mostly_empty)
    for col in ("公告日期", "报告日"):
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def _statement_report(
    ticker: str,
    statement_symbol: str,
    fallback_kind: str,
    label: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    symbol = _require_a_share(ticker)
    errors = []
    try:
        df = _fetch_akshare_statement(symbol, statement_symbol, curr_date, freq=freq)
        source = f"AKShare/Sina {statement_symbol}"
    except Exception as exc:  # noqa: BLE001 - BaoStock fallback may still work
        errors.append(f"AKShare: {exc}")
        df = _fetch_baostock_financial_rows(symbol, fallback_kind, curr_date, freq=freq)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
        source = f"BaoStock {fallback_kind}"

    header = (
        f"# {label} for {symbol.display} ({freq})\n"
        f"# Source: {source}; rows are filtered to announcements available by "
        f"{_ymd(_date(curr_date))}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    if errors:
        header += f"# Fallback notes: {'; '.join(errors)}\n\n"
    return header + df.to_csv(index=False)


def get_china_balance_sheet(
    ticker: Annotated[str, "China A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement_report(ticker, "资产负债表", "balance", "Balance Sheet", freq, curr_date)


def get_china_cashflow(
    ticker: Annotated[str, "China A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement_report(ticker, "现金流量表", "cashflow", "Cash Flow", freq, curr_date)


def get_china_income_statement(
    ticker: Annotated[str, "China A-share ticker symbol"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    return _statement_report(ticker, "利润表", "profit", "Income Statement", freq, curr_date)


def _latest_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_china_fundamentals(
    ticker: Annotated[str, "China A-share ticker symbol"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    symbol = _require_a_share(ticker)
    as_of = _ymd(_date(curr_date))

    sections: list[str] = [
        f"# China A-share Fundamentals for {symbol.display} (requested: {ticker})",
        f"# Rows are filtered to data available on or before {as_of}",
        "",
    ]

    identity: dict[str, str] = {"Ticker": symbol.display, "Yahoo Form": symbol.yahoo}
    with contextlib.suppress(Exception):
        identity.update({k: v for k, v in _fetch_baostock_basic(symbol).items() if v})
    with contextlib.suppress(Exception):
        identity.update({k: v for k, v in _fetch_akshare_static_info(symbol).items() if v})

    sections += ["## Identity", ""]
    for key, value in identity.items():
        sections.append(f"- {key}: {value}")

    valuation: dict[str, str] = {}
    with contextlib.suppress(Exception):
        valuation.update(_latest_baostock_valuation(symbol, curr_date))
    with contextlib.suppress(Exception):
        valuation.update(_fetch_tushare_daily_basic(symbol, curr_date))
    if valuation:
        sections += ["", "## Valuation", ""]
        for key, value in valuation.items():
            sections.append(f"- {key}: {value}")

    financial_rows: dict[str, dict] = {}
    for kind, title in (("profit", "Profitability"), ("balance", "Balance Risk"), ("cashflow", "Cash Flow Quality")):
        with contextlib.suppress(Exception):
            financial_rows[title] = _latest_row(
                _fetch_baostock_financial_rows(symbol, kind, curr_date, limit=1)
            )

    if financial_rows:
        sections += ["", "## Latest Announced Financial Ratios", ""]
        key_map = {
            "Profitability": [
                ("pubDate", "Announcement Date"),
                ("statDate", "Report Date"),
                ("roeAvg", "ROE Avg"),
                ("npMargin", "Net Profit Margin"),
                ("gpMargin", "Gross Profit Margin"),
                ("netProfit", "Net Profit"),
                ("epsTTM", "EPS TTM"),
                ("MBRevenue", "Main Business Revenue"),
            ],
            "Balance Risk": [
                ("pubDate", "Announcement Date"),
                ("statDate", "Report Date"),
                ("currentRatio", "Current Ratio"),
                ("quickRatio", "Quick Ratio"),
                ("cashRatio", "Cash Ratio"),
                ("liabilityToAsset", "Liability To Asset"),
                ("assetToEquity", "Asset To Equity"),
            ],
            "Cash Flow Quality": [
                ("pubDate", "Announcement Date"),
                ("statDate", "Report Date"),
                ("CFOToOR", "Operating Cash Flow / Revenue"),
                ("CFOToNP", "Operating Cash Flow / Net Profit"),
                ("CFOToGr", "Operating Cash Flow / Gross Revenue"),
            ],
        }
        for title, row in financial_rows.items():
            sections += [f"### {title}", ""]
            for raw_key, label in key_map[title]:
                if raw_key in row and row[raw_key] not in ("", None):
                    sections.append(f"- {label}: {_fmt(row[raw_key], 4)}")
            sections.append("")

    if len(sections) <= 5:
        raise NoMarketDataError(ticker, symbol.display, "no China fundamentals fields returned")
    return "\n".join(sections).strip()


def _filter_news_window(df: pd.DataFrame, date_col: str, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", format="mixed")
    start = _date(start_date)
    end_exclusive = _date(end_date) + pd.Timedelta(days=1)
    out = out[(out[date_col] >= start) & (out[date_col] < end_exclusive)]
    return out.sort_values(date_col, ascending=False).reset_index(drop=True)


def _eastmoney_news(symbol: AShareSymbol, start_date: str, end_date: str, limit: int) -> list[str]:
    ak = _import_akshare()
    raw = ak.stock_news_em(symbol=symbol.code)
    news = _filter_news_window(raw, "发布时间", start_date, end_date).head(limit)
    lines = []
    for _, row in news.iterrows():
        title = _truncate(row.get("新闻标题"), 160)
        source = _truncate(row.get("文章来源"), 80) or "Eastmoney"
        when = _ymd(row.get("发布时间"))
        summary = _truncate(row.get("新闻内容"), 420)
        link = _truncate(row.get("新闻链接"), 240)
        item = f"### {title} (source: {source}, date: {when})\n"
        if summary:
            item += summary + "\n"
        if link:
            item += f"Link: {link}\n"
        lines.append(item)
    return lines


def _eastmoney_commentary(symbol: AShareSymbol, start_date: str, end_date: str) -> list[str]:
    ak = _import_akshare()
    blocks: list[str] = []

    with contextlib.suppress(Exception):
        raw = _quiet_call(ak.stock_comment_em)
        if raw is not None and not raw.empty and {"代码", "交易日"}.issubset(raw.columns):
            raw["代码"] = raw["代码"].astype(str).str.zfill(6)
            filtered = _filter_news_window(raw[raw["代码"] == symbol.code], "交易日", start_date, end_date)
            if not filtered.empty:
                row = filtered.iloc[0]
                blocks.append(
                    "- Eastmoney stock-comment snapshot "
                    f"({row.get('交易日')}): score={_fmt(row.get('综合得分'))}, "
                    f"attention_index={_fmt(row.get('关注指数'))}, "
                    f"institution_participation={_fmt(row.get('机构参与度'))}, "
                    f"main_cost={_fmt(row.get('主力成本'))}."
                )

    detail_calls = [
        ("user attention", "交易日", "用户关注指数", ak.stock_comment_detail_scrd_focus_em),
        ("participation desire", "交易日期", "参与意愿", ak.stock_comment_detail_scrd_desire_em),
        ("composite score", "交易日", "评分", ak.stock_comment_detail_zhpj_lspf_em),
    ]
    for label, date_col, value_col, fn in detail_calls:
        with contextlib.suppress(Exception):
            raw = _quiet_call(fn, symbol=symbol.code)
            filtered = _filter_news_window(raw, date_col, start_date, end_date).head(5)
            if not filtered.empty:
                pairs = [
                    f"{_ymd(row[date_col])}: {_fmt(row.get(value_col))}"
                    for _, row in filtered.iterrows()
                ]
                blocks.append(f"- Eastmoney {label}: " + "; ".join(pairs))
    return blocks


def get_china_news(
    ticker: Annotated[str, "China A-share ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    symbol = _require_a_share(ticker)
    limit = int(get_config().get("news_article_limit", 20))
    sections = [
        f"## China A-share News and Sentiment for {symbol.display}, from {start_date} to {end_date}",
        "",
    ]

    news_lines = _eastmoney_news(symbol, start_date, end_date, limit)
    if news_lines:
        sections += ["## Eastmoney company news", "", *news_lines]

    commentary = _eastmoney_commentary(symbol, start_date, end_date)
    if commentary:
        sections += [
            "",
            "## Eastmoney retail/sentiment proxies",
            "",
            *commentary,
            "",
            (
                "These are platform-derived attention/participation/score metrics, "
                "not direct buy/sell recommendations."
            ),
        ]

    if len(sections) <= 2:
        return (
            f"No China A-share news or Eastmoney sentiment proxy data found for "
            f"{symbol.display} between {start_date} and {end_date}."
        )
    return "\n".join(sections).strip()


def get_china_global_news(
    curr_date: Annotated[str, "current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "number of days to look back"] = None,
    limit: Annotated[int, "maximum number of articles"] = None,
) -> str:
    curr = _date(curr_date)
    today = pd.Timestamp.today().normalize()
    if abs((today - curr).days) > 2:
        raise NoMarketDataError(
            "china_global_news",
            "china_global_news",
            "AKShare China macro headline endpoint has no reliable publish dates for historical filtering",
        )
    if limit is None:
        limit = int(get_config().get("global_news_article_limit", 10))
    ak = _import_akshare()
    raw = ak.stock_news_main_cx()
    if raw is None or raw.empty:
        raise NoMarketDataError("china_global_news", "china_global_news", "no China macro headlines")
    lines = [
        f"## Latest China market/macro headlines for {_ymd(curr)}",
        "",
        "Source: AKShare/Caixin latest-news endpoint. This endpoint does not expose publish dates, "
        "so it is used only for near-live runs.",
        "",
    ]
    for _, row in raw.head(int(limit)).iterrows():
        tag = _truncate(row.get("tag"), 80)
        summary = _truncate(row.get("summary"), 420)
        url = _truncate(row.get("url"), 240)
        lines.append(f"### {tag or 'China market news'}")
        if summary:
            lines.append(summary)
        if url:
            lines.append(f"Link: {url}")
        lines.append("")
    return "\n".join(lines).strip()


def get_china_insider_transactions(ticker: Annotated[str, "China A-share ticker symbol"]) -> str:
    symbol = _require_a_share(ticker)
    return (
        f"No standardized China A-share insider-transaction feed is configured for "
        f"{symbol.display}. For A-share experiments, use exchange announcements and "
        "shareholder-change datasets as a separate future enrichment; do not infer "
        "insider activity from missing data."
    )
