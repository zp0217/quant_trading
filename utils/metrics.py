"""
Performance Metric computation utility file

Total return, CAGR, Maximum Drawdown (MDD), Sharpe ratio, etc. 
"""

import pandas as pd
import numpy as np


def calculate_strategy_metrics(returns: pd.Series, strategy_name: str = "") -> dict:
    """
    Calculate key performance metrics from a strategy return series

    Args:
        returns: Daily return series
        strategy_name: Strategy name

    Returns:
        dict: Performance metrics
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    # Basic statistics
    total_return    = (1 + returns).prod() - 1
    n_years         = len(returns) / 252
    cagr            = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    annual_vol      = returns.std() * np.sqrt(252)
    sharpe          = (returns.mean() * 252) / (annual_vol + 1e-9)

    # Maximum Drawdown (MDD)
    cumulative      = (1 + returns).cumprod()
    rolling_max     = cumulative.cummax()
    drawdown        = (cumulative - rolling_max) / (rolling_max + 1e-9)
    mdd             = drawdown.min()

    # Win rate
    win_rate        = (returns > 0).sum() / max(len(returns[returns != 0]), 1)

    # Profit Factor
    gains           = returns[returns > 0].sum()
    losses          = abs(returns[returns < 0].sum())
    profit_factor   = gains / max(losses, 1e-9)

    # Sortino Ratio (based on downside volatility)
    downside_std    = returns[returns < 0].std() * np.sqrt(252)
    sortino         = (returns.mean() * 252) / max(downside_std, 1e-9)

    # Trade count (based on position changes)
    trades = (returns != 0).sum()

    return {
        "strategy": strategy_name,
        "total_return":    round(total_return * 100, 2),    # %
        "cagr":            round(cagr * 100, 2),             # %
        "annual_vol":      round(annual_vol * 100, 2),       # %
        "sharpe_ratio":    round(sharpe, 3),
        "sortino_ratio":   round(sortino, 3),
        "mdd":             round(mdd * 100, 2),              # %
        "win_rate":        round(win_rate * 100, 2),         # %
        "profit_factor":   round(profit_factor, 3),
        "active_days":     trades,
    }


def compare_strategies(strategy_results: list) -> pd.DataFrame:
    """Generate comparison table for multiple strategies"""
    records = []
    for result in strategy_results:
        m = result.get("metrics", {})
        m["strategy"] = result.get("name", "Unknown")
        records.append(m)
    df = pd.DataFrame(records)
    if "strategy" in df.columns:
        df = df.set_index("strategy")
    return df
