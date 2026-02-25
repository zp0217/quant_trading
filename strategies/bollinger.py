

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
        "BB_Upper": np.nan, "BB_Lower": np.nan, "BB_PctB": 0.5,
    }, index=df.index if len(df) > 0 else pd.DatetimeIndex([]))
    return {
        "name": "Bollinger Bands",
        "data": empty,
        "buy_signals":  pd.DatetimeIndex([]),
        "sell_signals": pd.DatetimeIndex([]),
        "metrics": {},
        "params": {},
        "_empty": True,
    }


def run_bollinger_strategy(df: pd.DataFrame,
                            window: int = 20,
                            num_std: float = 2.0) -> dict:
    try:
        data = df.copy()
        data["BB_MA"]    = data["Close"].rolling(window).mean()
        bb_std           = data["Close"].rolling(window).std()
        data["BB_Upper"] = data["BB_MA"] + num_std * bb_std
        data["BB_Lower"] = data["BB_MA"] - num_std * bb_std
        data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (data["BB_MA"] + 1e-9)
        data.dropna(inplace=True)

        if len(data) < MIN_ROWS:
            return _empty_result(df)

        bb_width_mean = data["BB_Width"].rolling(50).mean()
        data["Signal"] = 0

        for i in range(1, len(data)):
            close  = data["Close"].iloc[i]
            upper  = data["BB_Upper"].iloc[i]
            lower  = data["BB_Lower"].iloc[i]
            ma     = data["BB_MA"].iloc[i]
            width  = data["BB_Width"].iloc[i]
            w_mean = bb_width_mean.iloc[i] if i >= 50 else width
            prev_signal = data["Signal"].iloc[i - 1]
            volatility_ok = width >= w_mean * 0.8

            if close < lower and volatility_ok:
                data.iloc[i, data.columns.get_loc("Signal")] = 1
            elif close > upper and volatility_ok:
                data.iloc[i, data.columns.get_loc("Signal")] = -1
            elif prev_signal == 1 and close >= ma:
                data.iloc[i, data.columns.get_loc("Signal")] = 0
            elif prev_signal == -1 and close <= ma:
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
        metrics      = calculate_strategy_metrics(data["Strategy_Return"], "Bollinger Bands")

        return {
            "name": "Bollinger Bands",
            "data": data,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "metrics": metrics,
            "params": {"window": window, "num_std": num_std},
        }
    except Exception:
        return _empty_result(df)
