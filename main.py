

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np




def run_cli_analysis(ticker: str, period: str, run_ml: bool = False):
    """CLI mode: Single ticker analysis"""
    from data.fetcher import fetch_stock_data, get_stock_info
    from strategies.momentum import run_momentum_strategy
    from strategies.bollinger import run_bollinger_strategy
    from strategies.rsi import run_rsi_strategy
    from utils.metrics import compare_strategies

    print(f"Fetching data: {ticker} ({period})")
    df = fetch_stock_data(ticker, period=period)
    if df.empty:
        print(f"Unable to fetch data: {ticker}")
        return

    info = get_stock_info(ticker)
    print(f"{info.get('name', ticker)} | {len(df)} trading days of data\n")

    # Run strategies
    print("Running strategy backtests...")
    results = [
        run_momentum_strategy(df),
        run_bollinger_strategy(df),
        run_rsi_strategy(df),
    ]

    comp = compare_strategies(results)
    print("\n" + "─"*60)
    print("Strategy Performance Comparison")
    print("─"*60)
    print(comp.to_string())
    print()

    # Current signals
    print("─"*60)
    print("Current Trading Signals")
    print("─"*60)
    for r in results:
        data   = r["data"]
        signal = data["Signal"].iloc[-1]
        sig_str = "BUY" if signal == 1 else ("SELL" if signal == -1 else "NEUTRAL")
        print(f"  {r['name']:25s} → {sig_str}")
    print()

    if run_ml:
        print("─"*60)
        print("Running ARIMA time series forecast...")
        print("─"*60)
        from models.arima_model import run_arima
        arima = run_arima(df, forecast_days=30)
        if "error" not in arima:
            print(f"  ARIMA Order:            {arima['order']}")
            print(f"  Current Price:          {arima['current_price']:,.4f}")
            print(f"  30-Day Forecast:        {arima['forecast_30d']:,.4f}")
            print(f"  Expected Return (30D):  {arima['expected_return_30d']:+.2f}%")
            print(f"  Direction:              {'↑ UP' if arima['direction'] == 'UP' else '↓ DOWN'}")
            print(f"  MAPE:                   {arima['metrics']['MAPE']}%")


def run_portfolio_analysis(tickers: list, period: str):
    """CLI mode: Portfolio analysis"""
    from data.fetcher import fetch_multiple_stocks
    from strategies.momentum import run_momentum_strategy
    from portfolio.optimizer import build_portfolio, get_signal_score

    print(f"Fetching data: {', '.join(tickers)}")
    data_dict = fetch_multiple_stocks(tickers, period=period)

    returns_dict = {}
    signal_scores = {}

    for ticker, df in data_dict.items():
        if df.empty:
            continue
        r = run_momentum_strategy(df)
        returns_dict[ticker] = r["data"]["Return_1d"]
        signal_scores[ticker] = get_signal_score([r])

    if len(returns_dict) < 2:
        print("At least 2 tickers are required.")
        return

    print("\n" + "─"*60)
    print("Portfolio Optimization Results")
    print("─"*60)

    for method in ["equal_weight", "min_variance", "max_sharpe"]:
        result = build_portfolio(returns_dict, method=method, signal_scores=signal_scores)
        labels = {"equal_weight": "Equal Weight", "min_variance": "Min Variance", "max_sharpe": "Max Sharpe"}
        print(f"\n  [{labels[method]}]")
        for t, w in result["weights"].items():
            print(f"    {t:15s}: {w:>6.2f}%")
        pm = result["portfolio_metrics"]
        print(f"    → Expected Return: {pm['expected_return']}%  |  Volatility: {pm['volatility']}%"
              f"  |  Sharpe: {pm['sharpe_ratio']}  |  MDD: {pm['mdd']}%")


def main():
   

    parser = argparse.ArgumentParser(description="AlgoTrading Analysis Platform")
    parser.add_argument("--dashboard", action="store_true", help="Launch Dash dashboard")
    parser.add_argument("--ticker",    type=str, help="Single ticker analysis (e.g., AAPL)")
    parser.add_argument("--portfolio", type=str, help="Portfolio analysis (e.g., AAPL,MSFT,TSLA)")
    parser.add_argument("--period",    type=str, default="2y", help="Period (1y/2y/3y/5y)")
    parser.add_argument("--ml",        action="store_true", help="Include ML predictions")
    args = parser.parse_args()

    if args.dashboard:
        from dashboard.app import app
        print("🌐 Launching dashboard... http://localhost:8050")
        app.run(debug=False, host="0.0.0.0", port=8050)

    elif args.portfolio:
        tickers = [t.strip().upper() for t in args.portfolio.split(",")]
        run_portfolio_analysis(tickers, args.period)

    elif args.ticker:
        run_cli_analysis(args.ticker.upper(), args.period, run_ml=args.ml)

    else:
        print("Usage:")
        print("  Dashboard:   python main.py --dashboard")
        print("  CLI Analysis: python main.py --ticker AAPL --period 2y --ml")
        print("  Portfolio:   python main.py --portfolio AAPL,MSFT,TSLA --period 2y")
        print()
        print("Launching dashboard...")
        from dashboard.app import app
        app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
