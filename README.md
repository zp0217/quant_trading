# quant_trading 

<div align="center">

**Algorithmic trading analysis**

*Three trading strategies · Three ML/DL forecasting models · Portfolio optimization · Interactive web dashboard*

## Dashboard Screenshots

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./photo1.png" alt="Chart tab" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="./photo2.png" alt="Risk analytics tab" width="100%" />
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="./photo3.png" alt="ML forecast tab" width="100%" />
    </td>
    <td width="50%" align="center">
      <img src="./photo4.png" alt="Overview tab" width="100%" />
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" colspan="2">
      <img src="./photo5.png" alt="Portfolio optimizer tab" width="70%" />
    </td>
  </tr>
</table>

## Overview

this project is an  quantitative trading analysis demo  that integrates **technical analysis**, **strategy backtesting**, **machine learning forecasting**, and **portfolio optimization** into a interactive web dashboard.

## Motivation

As a data science student, I see strong value in applying analytical skills across different domains to build practical, transferable expertise. Finance is one of the fields I am most interested in, and this project was motivated by my curiosity about how data science can be used to generate insights and support decision-making in financial contexts.

This project also aligns with my ongoing CFA exam preparation. It provides an opportunity to apply and validate my financial knowledge in a practical setting, combining what I have already learned with concepts I am currently studying through the CFA curriculum.


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Entry Point)                    │
│                   CLI Mode / Dashboard Mode                     │
└─────────┬──────────────────────────────────┬────────────────────┘
          │                                  │
          ▼                                  ▼
┌──────────────────┐              ┌─────────────────────┐
│  data/fetcher.py │              │  dashboard/app.py   │
│  ─────────────── │              │  ─────────────────  │
│  yfinance OHLCV  │              │  Dash + Plotly      │
│  Technical       │              │  7-Tab Terminal UI  │
│  Indicators      │              │  Clientside JS      │
│  (MA/BB/RSI/     │              │  Session Storage    │
│   MACD/ATR)      │              └──────────┬──────────┘
└────────┬─────────┘                         │
         │                                   │ renders
         ▼                                   │
┌──────────────────────────────┐             │
│     strategies/              │             │
│     ────────────             │             │
│  momentum.py  (Trend)        │◄────────────┤
│  bollinger.py (MeanRev)      │             │
│  rsi.py       (Contrarian)   │             │
└────────┬─────────────────────┘             │
         │                                   │
         ▼                                   │
┌──────────────────────────────┐             │
│     models/                  │             │
│     ────────                 │             │
│  lstm_model.py   (DL)        │◄────────────┤
│  gru_model.py    (DL)        │             │
│  arima_model.py  (Stats)     │             │
└────────┬─────────────────────┘             │
         │                                   │
         ▼                                   │
┌──────────────────────────────┐             │
│     portfolio/               │             │
│     ──────────               │             │
│  optimizer.py                │◄────────────┘
│  (EqualW/MinVar/MaxSharpe)   │
│  paper_trading.py            │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│     utils/metrics.py         │
│     ────────────────         │
│  Sharpe, MDD, CAGR, Win%    │
│  Sortino, Profit Factor      │
└──────────────────────────────┘
```

---

## Installation(MacOS)

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/algotrade-terminal.git
cd algotrade-terminal

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `yfinance` | ≥ 0.2.28 | Market data from Yahoo Finance |
| `pandas` | ≥ 1.5.0 | Data manipulation |
| `numpy` | ≥ 1.23.0 | Numerical computation |
| `scikit-learn` | ≥ 1.2.0 | MinMaxScaler for feature normalization |
| `tensorflow` | ≥ 2.12.0 | LSTM & GRU deep learning models |
| `statsmodels` | ≥ 0.14.0 | ARIMA time-series model |
| `pmdarima` | ≥ 2.0.0 | Auto-ARIMA order selection |
| `plotly` | ≥ 5.14.0 | Interactive charts |
| `dash` | ≥ 2.10.0 | Web application framework |
| `dash-bootstrap-components` | ≥ 1.4.0 | UI components |
| `scipy` | ≥ 1.10.0 | Portfolio optimization (SLSQP) |
| `matplotlib` | ≥ 3.7.0 | Supplementary plotting |
| `ta` | ≥ 0.10.2 | Technical analysis utilities |

---

## Usage

### Web Dashboard (Recommended)

```bash
python main.py --dashboard
# Open browser  http://localhost:8050
```

1. Type a ticker symbol (e.g., `AAPL`) in the input field
2. Click **ADD** (supports up to 5 tickers simultaneously)
3. Select ML model (`LSTM`, `GRU`, `ARIMA`, or `ALL`), period, and epochs
4. Click **▶ RUN** to start analysis
5. Navigate between 7 tabs to explore results


### Note for Windows Users

# Clone the repository
```bash

git clone https://github.com/yourusername/algotrade-terminal.git
cd algotrade-terminal

# (Optional) Create a virtual environment
py -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

```bash

python main.py --dashboard
# Open browser: http://localhost:8050

```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dashboard` | flag | — | Launch web dashboard at port 8050 |
| `--ticker` | string | — | Single ticker analysis (e.g., `AAPL`) |
| `--portfolio` | string | — | Comma-separated tickers (e.g., `AAPL,MSFT,TSLA`) |
| `--period` | string | `2y` | Data period: `1y`, `2y`, `3y`, `5y` |
| `--ml` | flag | — | Include ML model predictions in CLI mode |

---

## Project Structure

```
trading_final_version/
│
├── main.py                         # Entry point (CLI + Dashboard router)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── __init__.py
│   └── fetcher.py                  # Data acquisition + technical indicator computation
│       ├── fetch_stock_data()      #   → yfinance OHLCV download + indicator pipeline
│       ├── fetch_multiple_stocks() #   → Batch multi-ticker download
│       ├── get_stock_info()        #   → Fundamental info (market cap, PE, sector)
│       ├── _add_technical_indicators() # → MA, BB, RSI, MACD, ATR, returns, volume
│       └── _compute_rsi()          #   → Wilder-style RSI via EWM
│
├── strategies/
│   ├── __init__.py
│   ├── momentum.py                 # Strategy 1: Dual MA Crossover + ROC filter
│   │   └── run_momentum_strategy() #   → Vectorized signal generation
│   ├── bollinger.py                # Strategy 2: Bollinger Band Mean Reversion
│   │   └── run_bollinger_strategy()#   → Stateful loop with volatility filter
│   └── rsi.py                      # Strategy 3: RSI Mean Reversion + MA200 filter
│       └── run_rsi_strategy()      #   → Stateful loop with trend confirmation
│
├── models/
│   ├── __init__.py
│   ├── lstm_model.py               # LSTM deep learning forecaster
│   │   └── train_lstm()            #   → Adaptive lookback, MC Dropout CI
│   ├── gru_model.py                # GRU deep learning forecaster
│   │   └── train_gru()             #   → Lightweight alternative to LSTM
│   └── arima_model.py              # ARIMA statistical time-series model
│       ├── run_arima()             #   → Log-transform, auto-order, 4-level fallback
│       ├── _select_order()         #   → pmdarima auto_arima or manual AIC search
│       ├── _forecast()             #   → Forecast with CI, linear extrapolation fallback
│       └── _fit_arima()            #   → Multi-method fitting with cascading fallback
│
├── portfolio/
│   ├── __init__.py
│   ├── optimizer.py                # Portfolio optimization engine
│   │   ├── build_portfolio()       #   → Equal Weight / Min Variance / Max Sharpe
│   │   ├── efficient_frontier()    #   → 500 Monte Carlo simulated portfolios
│   │   └── get_signal_score()      #   → Strategy-signal-based weighting
│   └── paper_trading.py            # Paper trading simulation engine
│       └── run_paper_trading()     #   → Virtual $100K, 0.1% commission + slippage
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                  # Performance metrics calculator
│       ├── calculate_strategy_metrics()  # → Sharpe, MDD, CAGR, win rate, etc.
│       └── compare_strategies()          # → Multi-strategy comparison DataFrame
│
└── dashboard/
    ├── __init__.py
    └── app.py                      # Dash web dashboard (~1,200 lines)
        ├── Layout                  #   → Sidebar + Header + 7-tab content area
        ├── Callbacks               #   → Ticker management, analysis, tab routing
        ├── Tab Renderers           #   → Overview, Chart, Forecast, Universe,
        │                           #     Risk, Portfolio, Live Log
        └── Utilities               #   → Safe number conversion, chart helpers
```

---