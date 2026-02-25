

import pandas as pd
import numpy as np
from utils.metrics import calculate_strategy_metrics

MIN_ROWS = 5


def _empty_result(df: pd.DataFrame) -> dict:
    empty = pd.DataFrame({
        "Close": df["Close"] if "Close" in df.columns else [],
        "Signal": 0, "Position": 0,
        "Return_1d": 0.0, "Strategy_Return": 0.0, "Buy_Hold_Return": 0.0,
        "Cumulative_Strategy": 1.0, "Cumulative_BuyHold": 1.0,
        "RSI": 50.0,
    }, index=df.index if len(df) > 0 else pd.DatetimeIndex([]))
    return {
        "name": "RSI Mean Reversion",
        "data": empty,
        "buy_signals":  pd.DatetimeIndex([]),
        "sell_signals": pd.DatetimeIndex([]),
        "metrics": {},
        "params": {},
        "_empty": True,
    }


def run_rsi_strategy(df: pd.DataFrame,
                      rsi_period: int = 14,
                      oversold: float = 30.0,
                      overbought: float = 70.0) -> dict:
    try:
        data = df.copy()

        # RSI calculation
        delta    = data["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        data["RSI_Custom"] = 100 - (100 / (1 + rs))

        # MA200: use if sufficient data, otherwise disable filter
        has_ma200 = len(data) >= 210   # 200 + buffer
        if has_ma200:
            data["MA200"] = data["Close"].rolling(200).mean()
        else:
            data["MA200"] = data["Close"]   # Disable filter (always above)

        data.dropna(inplace=True)

        if len(data) < MIN_ROWS:
            return _empty_result(df)

        data["Signal"] = 0

        for i in range(1, len(data)):
            rsi         = data["RSI_Custom"].iloc[i]
            close       = data["Close"].iloc[i]
            ma200       = data["MA200"].iloc[i]
            prev_signal = data["Signal"].iloc[i - 1]
            above_ma200 = close > ma200

            if rsi < oversold and above_ma200:
                data.iloc[i, data.columns.get_loc("Signal")] = 1
            elif rsi > overbought:
                data.iloc[i, data.columns.get_loc("Signal")] = -1
            elif prev_signal == 1 and rsi >= 50:
                data.iloc[i, data.columns.get_loc("Signal")] = 0
            elif prev_signal == -1 and rsi <= 50:
                data.iloc[i, data.columns.get_loc("Signal")] = 0
            else:
                data.iloc[i, data.columns.get_loc("Signal")] = prev_signal

        data["Position"]            = data["Signal"].shift(1).fillna(0)
        data["Position"]            = data["Position"].replace(-1, 0)
        data["Strategy_Return"]     = data["Position"] * data["Return_1d"]
        data["Buy_Hold_Return"]     = data["Return_1d"]
        data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()
        data["Cumulative_BuyHold"]  = (1 + data["Buy_Hold_Return"]).cumprod()

        buy_signals  = data[(data["Signal"] == 1)  & (data["Signal"].shift(1) != 1)].index
        sell_signals = data[(data["Signal"] == -1) & (data["Signal"].shift(1) != -1)].index
        metrics      = calculate_strategy_metrics(data["Strategy_Return"], "RSI Mean Reversion")

        return {
            "name": "RSI Mean Reversion",
            "data": data,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "metrics": metrics,
            "params": {"rsi_period": rsi_period,
                       "oversold": oversold,
                       "overbought": overbought},
        }
    except Exception:
        return _empty_result(df)
