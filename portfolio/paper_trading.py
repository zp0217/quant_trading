"""
Investment simulation 

"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


INITIAL_CASH = 10000.0          # Initial cash (USD)
COMMISSION   = 0.001               # Trading commission 0.1%
SLIPPAGE     = 0.001               # Slippage 0.1%


@dataclass
class Trade:
    date:       object
    ticker:     str
    side:       str          # 'BUY' | 'SELL'
    strategy:   str
    price:      float
    shares:     float
    commission: float
    pnl:        float = 0.0
    cum_pnl:    float = 0.0


def run_paper_trading(df: pd.DataFrame,
                      strategy_results: list,
                      ticker: str,
                      initial_cash: float = INITIAL_CASH) -> dict:
    """
    investing simulation based on strategy signals

    Args:
        df: OHLCV DataFrame
        strategy_results: [momentum, bollinger, rsi] result list
        ticker: Stock symbol
        initial_cash: Initial virtual capital

    Returns:
        dict: trades, portfolio_history, summary
    """
    strategy_names = ["Momentum", "Bollinger", "RSI"]
    all_trades     = []
    portfolio_hist = {}      # strategy -> equity curve
    summaries      = {}

    for result, s_name in zip(strategy_results, strategy_names):
        data   = result["data"].copy()
        trades, equity = _simulate_strategy(data, ticker, s_name, initial_cash)
        all_trades.extend(trades)
        portfolio_hist[s_name] = equity
        summaries[s_name]      = _calc_paper_summary(trades, equity, initial_cash)

    # All trades DataFrame
    if all_trades:
        trade_df = pd.DataFrame([
            {
                "Date":       str(t.date)[:10],
                "Ticker":     t.ticker,
                "Strategy":   t.strategy,
                "Side":       t.side,
                "Exec Price": round(t.price, 4),
                "Shares":     round(t.shares, 4),
                "Commission": round(t.commission, 2),
                "P&L":        round(t.pnl, 2),
                "Cum P&L":    round(t.cum_pnl, 2),
            }
            for t in all_trades
        ])
    else:
        trade_df = pd.DataFrame()

    return {
        "trades":       all_trades,
        "trade_df":     trade_df,
        "equity_curves": portfolio_hist,
        "summaries":    summaries,
    }


def _simulate_strategy(data: pd.DataFrame, ticker: str,
                        s_name: str, initial_cash: float):
    
    cash     = initial_cash
    shares   = 0.0
    entry_px = 0.0
    trades   = []
    equity   = []
    cum_pnl  = 0.0

    for i in range(1, len(data)):
        date    = data.index[i]
        price   = float(data["Open"].iloc[i])    # Execute at next day's open
        sig     = int(data["Signal"].iloc[i - 1])
        prev_sig = int(data["Signal"].iloc[i - 2]) if i >= 2 else 0

        # Buy entry
        if sig == 1 and prev_sig != 1 and shares == 0:
            exec_px = price * (1 + SLIPPAGE)
            max_shares = (cash * 0.95) / exec_px   # Use 95% of capital
            if max_shares > 0.01:
                commission  = exec_px * max_shares * COMMISSION
                shares      = max_shares
                cash       -= exec_px * shares + commission
                entry_px    = exec_px
                trades.append(Trade(
                    date=date, ticker=ticker, side="BUY", strategy=s_name,
                    price=round(exec_px, 4), shares=round(shares, 4),
                    commission=round(commission, 2)
                ))

        # Sell exit
        elif (sig != 1 or sig == -1) and prev_sig == 1 and shares > 0:
            exec_px    = price * (1 - SLIPPAGE)
            commission = exec_px * shares * COMMISSION
            pnl        = (exec_px - entry_px) * shares - commission
            cum_pnl   += pnl
            cash      += exec_px * shares - commission
            trades.append(Trade(
                date=date, ticker=ticker, side="SELL", strategy=s_name,
                price=round(exec_px, 4), shares=round(shares, 4),
                commission=round(commission, 2),
                pnl=round(pnl, 2), cum_pnl=round(cum_pnl, 2)
            ))
            shares  = 0.0
            entry_px = 0.0

        # Net Asset Value
        close = float(data["Close"].iloc[i])
        nav   = cash + shares * close
        equity.append({"date": date, "nav": nav,
                        "cash": cash, "shares": shares, "price": close})

    equity_df = pd.DataFrame(equity).set_index("date") if equity else pd.DataFrame()
    return trades, equity_df


def _calc_paper_summary(trades: list, equity_df: pd.DataFrame, initial_cash: float) -> dict:
    """Paper trading performance summary"""
    if equity_df.empty:
        return {}

    final_nav    = float(equity_df["nav"].iloc[-1])
    total_return = (final_nav - initial_cash) / initial_cash * 100

    n_days       = len(equity_df)
    n_years      = n_days / 252
    ann_return   = ((final_nav / initial_cash) ** (1 / max(n_years, 0.01)) - 1) * 100

    daily_ret    = equity_df["nav"].pct_change().dropna()
    vol          = daily_ret.std() * np.sqrt(252) * 100
    sharpe       = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252) + 1e-9)

    roll_max     = equity_df["nav"].cummax()
    drawdown     = (equity_df["nav"] - roll_max) / roll_max
    mdd          = drawdown.min() * 100

    sell_trades  = [t for t in trades if t.side == "SELL"]
    wins         = [t for t in sell_trades if t.pnl > 0]
    #win_rate     = len(wins) / max(len(sell_trades), 1) * 100
    gain         = sum(t.pnl for t in wins)
    loss         = abs(sum(t.pnl for t in sell_trades if t.pnl <= 0))
    pf           = gain / max(loss, 1e-9)

    return {
        "initial_cash":  initial_cash,
        "final_nav":     round(final_nav, 2),
        "total_return":  round(total_return, 2),
        "ann_return":    round(ann_return, 2),
        "sharpe":        round(sharpe, 3),
        "mdd":           round(mdd, 2),
        #"win_rate":      round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "n_trades":      len(sell_trades),
        "total_pnl":     round(sum(t.pnl for t in sell_trades), 2),
        "current_cash":  round(float(equity_df["cash"].iloc[-1]), 2),
        "current_shares":round(float(equity_df["shares"].iloc[-1]), 4),
    }
