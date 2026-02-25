"""
Data Fetcher Module
- Stock data fetching via yfinance
- Automatic calculation of technical indicators (RSI, MACD, Bollinger Bands, etc.)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def fetch_stock_data(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and calculate technical indicators

    Args:
        ticker: Stock symbol (e.g., 'AAPL', '005930.KS')
        period: Period (1mo, 3mo, 6mo, 1y, 2y, 3y, 5y, 10y, ytd, max)
        interval: Interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)

    Returns:
        DataFrame: Includes OHLCV + technical indicator columns
    """
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)
        df = _add_technical_indicators(df)
        return df

    except Exception as e:
        print(f"[ERROR] fetch_stock_data({ticker}): {e}")
        return pd.DataFrame()


def fetch_multiple_stocks(tickers: list, period: str = "3y") -> dict:
    """Fetch multiple tickers simultaneously"""
    return {ticker: fetch_stock_data(ticker, period) for ticker in tickers}


def get_stock_info(ticker: str) -> dict:
    """Fetch stock fundamental info (market cap, P/E ratio, dividend yield, etc.)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        print(f"[WARN] get_stock_info({ticker}): {e}")
        return {"name": ticker}


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators (pure pandas/numpy, no ta library)"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Moving Averages ──────────────────────────────────────
    df["MA10"]  = close.rolling(10).mean()
    df["MA20"]  = close.rolling(20).mean()
    df["MA50"]  = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # ── Bollinger Bands (20-day, 2σ) ─────────────────────────
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * bb_std
    df["BB_Lower"] = df["MA20"] - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["MA20"]
    # %B: 0=lower band, 1=upper band
    df["BB_PctB"]  = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ── RSI (14-day) ─────────────────────────────────────────
    df["RSI"] = _compute_rsi(close, period=14)

    # ── MACD ─────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # ── ATR (14-day) ─────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ── Returns ──────────────────────────────────────────────
    df["Return_1d"]  = close.pct_change()
    df["Return_5d"]  = close.pct_change(5)
    df["Return_20d"] = close.pct_change(20)

    # ── Volume Moving Average ────────────────────────────────
    df["Vol_MA20"] = volume.rolling(20).mean()
    df["Vol_Ratio"] = volume / df["Vol_MA20"]

    # Only dropna on essential OHLCV columns (allow NaN for long rolling columns like MA200)
    return df.dropna(subset=["Open","High","Low","Close","Volume"])


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi
