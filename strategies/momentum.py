""""
Logic:
  Short MA(10) > Long MA(50) + ROC > 0 : Buy (Golden Cross)
  Short MA(10) < Long MA(50)           : Sell (Death Cross)
"""

import pandas as pd
import numpy as np
from utils.metrics import calculate_strategy_metrics

MIN_ROWS = 5   # Minimum required rows after dropna(set max 5 stocks are allowed for input)


def _empty_result(name: str, df: pd.DataFrame) -> dict:
    """Return empty result when data is insufficient (crash prevention)"""
    empty = pd.DataFrame({
        "Close": df["Close"] if "Close" in df.columns else [],
        "Signal": 0,
        "Position": 0,
        "Return_1d": 0.0,
        "Strategy_Return": 0.0,
        "Buy_Hold_Return": 0.0,
        "Cumulative_Strategy": 1.0,
        "Cumulative_BuyHold": 1.0,
    }, index=df.index if len(df) > 0 else pd.DatetimeIndex([]))
    return {
        "name": name,
        "data": empty,
        "buy_signals":  pd.DatetimeIndex([]),
        "sell_signals": pd.DatetimeIndex([]),
        "metrics": {},
        "params": {},
        "_empty": True,
    }


def run_momentum_strategy(df: pd.DataFrame,
                           short_window: int = 10,
                           long_window: int = 50,
                           momentum_window: int = 20) -> dict:
    try:
        data = df.copy()
        data["MA_Short"] = data["Close"].rolling(short_window).mean()
        data["MA_Long"]  = data["Close"].rolling(long_window).mean()
        data["ROC"]      = data["Close"].pct_change(momentum_window)
        data.dropna(inplace=True)

        if len(data) < MIN_ROWS:
            return _empty_result("Simple Momentum", df)

        data["Signal"] = 0
        bullish = (data["MA_Short"] > data["MA_Long"]) & (data["ROC"] > 0)
        bearish = (data["MA_Short"] < data["MA_Long"])
        data.loc[bullish, "Signal"] = 1
        data.loc[bearish, "Signal"] = -1

        data["Position"]            = data["Signal"].shift(1).fillna(0)
        data["Position"]            = data["Position"].replace(-1, 0)
        data["Strategy_Return"]     = data["Position"] * data["Return_1d"]
        data["Buy_Hold_Return"]     = data["Return_1d"]
        data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()
        data["Cumulative_BuyHold"]  = (1 + data["Buy_Hold_Return"]).cumprod()

        buy_signals  = data[data["Signal"].diff() == 1].index
        sell_signals = data[data["Signal"].diff() == -2].index
        metrics      = calculate_strategy_metrics(data["Strategy_Return"], "Momentum")

        return {
            "name": "Simple Momentum",
            "data": data,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "metrics": metrics,
            "params": {"short_window": short_window,
                       "long_window": long_window,
                       "momentum_window": momentum_window},
        }
    except Exception as e:
        return _empty_result("Simple Momentum", df)
