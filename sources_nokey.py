#!/usr/bin/env python3
"""
sources_nokey.py — No-key daily OHLCV fetchers and normalizers
Sources:
  - CoinMarketCap public data-api: /cryptocurrency/historical (quotes with OHLCV)
  - CryptoCompare: /data/v2/histoday (OHLCV)
  - CoinGecko: /coins/{id}/ohlc (OHLC) + /coins/{id}/market_chart (volume)
"""

from __future__ import annotations
import math
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List

# Config
ASSETS = ["BTC", "ETH", "USDT", "SOL", "XRP"]
QUOTE = "USD"

# Mappings
CMC_IDS = {
    "BTC": 1,
    "ETH": 1027,
    "USDT": 825,
    "SOL": 5426,
    "XRP": 52,
}
CG_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "SOL": "solana",
    "XRP": "ripple",
}

CMC_BASE = "https://api.coinmarketcap.com/data-api/v3"
CC_BASE  = "https://min-api.cryptocompare.com"
CG_BASE  = "https://api.coingecko.com/api/v3"


def _get(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 30):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_cmc_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV from CMC public 'historical' endpoint and normalize."""
    coin_id = CMC_IDS[symbol]
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days + 2)  # little buffer
    url = f"{CMC_BASE}/cryptocurrency/historical"
    params = {
        "id": coin_id,
        "convertId": 2781,  # USD in CMC internal mapping
        "timeStart": int(start.timestamp()),
        "timeEnd": int(now.timestamp()),
    }
    data = _get(url, params=params)
    quotes = data.get("data", {}).get("quotes", [])
    rows = []
    for q in quotes:
        d = q.get("timeOpen", "")[:10]
        quote = q.get("quote", {}) or {}
        rows.append({
            "date": d,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "source": "CMC",
            "symbol": symbol,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")
    return df


def fetch_cc_histoday(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV from CryptoCompare histoday (volumeTo as USD volume)."""
    url = f"{CC_BASE}/data/v2/histoday"
    params = {"fsym": symbol, "tsym": QUOTE, "limit": days}
    data = _get(url, params=params)
    arr = data.get("Data", {}).get("Data", [])
    rows = []
    for item in arr:
        ts = item.get("time")
        d = datetime.utcfromtimestamp(ts).date().isoformat() if ts else None
        rows.append({
            "date": d,
            "open": item.get("open"),
            "high": item.get("high"),
            "low": item.get("low"),
            "close": item.get("close"),
            "volume": item.get("volumeto"),  # USD volume
            "source": "CC",
            "symbol": symbol,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")
    return df


def fetch_cg_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLC from CoinGecko and merge with daily total volume."""
    cid = CG_IDS[symbol]
    # OHLC (candles): list of [timestamp(ms), open, high, low, close]
    url_ohlc = f"{CG_BASE}/coins/{cid}/ohlc"
    params_ohlc = {"vs_currency": QUOTE.lower(), "days": days}
    ohlc = _get(url_ohlc, params=params_ohlc)

    # Volume: market_chart total_volumes -> list of [timestamp(ms), volume]
    url_mc = f"{CG_BASE}/coins/{cid}/market_chart"
    params_mc = {"vs_currency": QUOTE.lower(), "days": days, "interval": "daily"}
    mc = _get(url_mc, params=params_mc)
    vol_series = mc.get("total_volumes", [])

    vol_map = {}
    for tms, vol in vol_series:
        d = datetime.utcfromtimestamp(tms / 1000).date().isoformat()
        vol_map[d] = vol

    rows = []
    for row in ohlc:
        tms, o, h, l, c = row
        d = datetime.utcfromtimestamp(tms / 1000).date().isoformat()
        rows.append({
            "date": d,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol_map.get(d),
            "source": "CG",
            "symbol": symbol,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["close"]).drop_duplicates(subset=["date"], keep="last")
    return df


def union_three_sources(symbol: str, days: int = 90) -> pd.DataFrame:
    dfs = []
    try:
        dfs.append(fetch_cmc_ohlcv(symbol, days))
    except Exception:
        pass
    time.sleep(0.2)
    try:
        dfs.append(fetch_cc_histoday(symbol, days))
    except Exception:
        pass
    time.sleep(0.2)
    try:
        dfs.append(fetch_cg_ohlcv(symbol, days))
    except Exception:
        pass
    if not dfs:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","source","symbol"])
    return pd.concat(dfs, ignore_index=True)


def fuse_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Median fuse per (symbol, date) for close & volume; max/min for highs/lows."""
    if df.empty:
        return df.copy()

    def _median(series):
        vals = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        if n % 2:
            return vals[n//2]
        return (vals[n//2 - 1] + vals[n//2]) / 2.0

    def _max(series):
        vals = [v for v in series if v is not None]
        return max(vals) if vals else None

    def _min(series):
        vals = [v for v in series if v is not None]
        return min(vals) if vals else None

    agg = df.groupby(["symbol","date"]).agg(
        open=("open", _median),
        high=("high", _max),
        low=("low", _min),
        close=("close", _median),
        volume=("volume", _median),
        sources=("source", "nunique"),
    ).reset_index()
    return agg


def dispersion_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    def _med(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        if n % 2: return vals[n//2]
        return (vals[n//2-1] + vals[n//2]) / 2.0

    out = []
    for (sym, d), sub in df.groupby(["symbol","date"]):
        prices = sub["close"].tolist()
        vols   = sub["volume"].tolist()
        def pct_spread(values):
            v = [x for x in values if x is not None and x > 0]
            if len(v) < 2: return 0.0 if v else None
            med = _med(v)
            return ((max(v) - min(v)) / med * 100.0) if med else None
        out.append({
            "symbol": sym, "date": d,
            "price_spread_pct": pct_spread(prices),
            "volume_spread_pct": pct_spread(vols),
            "sources": sub["source"].nunique(),
        })
    return pd.DataFrame(out)
