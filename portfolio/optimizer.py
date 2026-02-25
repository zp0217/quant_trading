"""
Portfolio Optimization Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method 1: Equal Weight
  - Equal allocation across all tickers
  - Simple but achieves diversification

Method 2: Minimum Variance
  - Minimize portfolio variance using scipy.optimize
  - Reflects inter-asset correlations via covariance matrix

Method 3: Maximum Sharpe Ratio
  - Portfolio with highest risk-adjusted return
  - Risk-free rate assumption: 4% annual (US benchmark)

Portfolio construction considerations:
  - Incorporate signal strength (Signal Score) from each strategy
  - Prefer ticker combinations with low correlation
  - Maximum single ticker weight: 40% cap
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


RISK_FREE_RATE = 0.04 / 252   # Daily risk-free rate - can be changed, this is also simulation


def build_portfolio(returns_dict: dict,
                    method: str = "max_sharpe",
                    signal_scores: Optional[dict] = None) -> dict:
    """
    Multi-asset portfolio optimization

    Args:
        returns_dict: {ticker: pd.Series of daily returns}
        method: 'equal_weight', 'min_variance', 'max_sharpe'
        signal_scores: {ticker: float} composite strategy signal score (optional)

    Returns:
        dict: weights, metrics, correlation matrix
    """
    tickers = list(returns_dict.keys())
    n = len(tickers)

    if n == 0:
        return {}

    # Align returns DataFrame
    ret_df = pd.DataFrame(returns_dict).dropna()
    if ret_df.empty or len(ret_df) < 30:
        return {"error": "Insufficient data for portfolio optimization"}

    # Covariance matrix, mean returns
    cov_matrix = ret_df.cov().values * 252           # Annualized
    mean_returns = ret_df.mean().values * 252         # Annualized

    # ── Optimization ─────────────────────────────────────────
    if method == "equal_weight":
        weights = np.array([1.0 / n] * n)

    elif method == "min_variance":
        weights = _min_variance(cov_matrix, n)

    elif method == "max_sharpe":
        weights = _max_sharpe(mean_returns, cov_matrix, n)

    else:
        weights = np.array([1.0 / n] * n)

    # Adjust weights by signal scores (optional)
    if signal_scores:
        scores = np.array([max(signal_scores.get(t, 0.5), 0.01) for t in tickers])
        score_weights = scores / scores.sum()
        # 50% optimization + 50% signal-based
        weights = 0.5 * weights + 0.5 * score_weights
        weights = weights / weights.sum()

    # ── Portfolio metric calculation ─────────────────────────
    port_return = np.dot(weights, mean_returns)
    port_vol    = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe      = (port_return - RISK_FREE_RATE * 252) / (port_vol + 1e-9)

    # Portfolio daily returns
    port_daily_returns = (ret_df * weights).sum(axis=1)
    cumulative = (1 + port_daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    mdd = ((cumulative - rolling_max) / rolling_max).min()

    # Correlation matrix
    corr_matrix = ret_df.corr()

    # Individual ticker metrics
    individual = {}
    for i, ticker in enumerate(tickers):
        individual[ticker] = {
            "weight": round(weights[i] * 100, 2),
            "expected_return": round(mean_returns[i] * 100, 2),
            "volatility": round(np.sqrt(cov_matrix[i, i]) * 100, 2),
            "sharpe": round((mean_returns[i] - RISK_FREE_RATE * 252) /
                            (np.sqrt(cov_matrix[i, i]) + 1e-9), 3)
        }

    return {
        "method": method,
        "tickers": tickers,
        "weights": {t: round(w * 100, 2) for t, w in zip(tickers, weights)},
        "portfolio_metrics": {
            "expected_return": round(port_return * 100, 2),
            "volatility":       round(port_vol * 100, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "mdd":              round(mdd * 100, 2),
        },
        "individual": individual,
        "correlation_matrix": corr_matrix,
        "daily_returns": port_daily_returns,
    }


def _min_variance(cov_matrix: np.ndarray, n: int) -> np.ndarray:
    """Minimum variance portfolio"""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.0, 0.4) for _ in range(n))    # Max 40%
    w0 = np.array([1.0 / n] * n)

    result = minimize(
        lambda w: np.dot(w, np.dot(cov_matrix, w)),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else w0


def _max_sharpe(mean_returns: np.ndarray,
                cov_matrix: np.ndarray,
                n: int) -> np.ndarray:
    """Maximum Sharpe ratio portfolio"""
    rf = RISK_FREE_RATE * 252
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.0, 0.4) for _ in range(n))
    w0 = np.array([1.0 / n] * n)

    def neg_sharpe(w):
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        return -(ret - rf) / (vol + 1e-9)

    result = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else w0


def efficient_frontier(returns_dict: dict, n_portfolios: int = 400) -> dict:
    """
    Generate efficient frontier
    - Simulate n random-weight portfolios
    - Return Max Sharpe / Min Vol points

    Returns:
        dict: vols, rets, sharpes, weights_list, special(max_sharpe, min_vol)
    """
    tickers = list(returns_dict.keys())
    n = len(tickers)
    if n < 2:
        return {"error": "At least 2 tickers are required"}

    ret_df = pd.DataFrame(returns_dict).dropna()
    if len(ret_df) < 30:
        return {"error": "Insufficient data"}

    cov_matrix  = ret_df.cov().values * 252
    mean_returns = ret_df.mean().values * 252
    rf = RISK_FREE_RATE * 252

    vols, rets, sharpes, weights_list = [], [], [], []

    np.random.seed(42)
    for _ in range(n_portfolios):
        # Dirichlet distribution for natural weight sampling
        w = np.random.dirichlet(np.ones(n))
        r = np.dot(w, mean_returns)
        v = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
        s = (r - rf) / (v + 1e-9)
        vols.append(round(v * 100, 4))
        rets.append(round(r * 100, 4))
        sharpes.append(round(s, 4))
        weights_list.append([round(x * 100, 2) for x in w])

    # Special points: Max Sharpe, Min Vol
    idx_ms  = int(np.argmax(sharpes))
    idx_mv  = int(np.argmin(vols))

    # Optimization-based Max Sharpe
    ms_weights = _max_sharpe(mean_returns, cov_matrix, n)
    ms_r = np.dot(ms_weights, mean_returns)
    ms_v = np.sqrt(np.dot(ms_weights, np.dot(cov_matrix, ms_weights)))

    # Optimization-based Min Vol
    mv_weights = _min_variance(cov_matrix, n)
    mv_r = np.dot(mv_weights, mean_returns)
    mv_v = np.sqrt(np.dot(mv_weights, np.dot(cov_matrix, mv_weights)))

    return {
        "tickers":      tickers,
        "vols":         vols,
        "rets":         rets,
        "sharpes":      sharpes,
        "weights_list": weights_list,
        "special": {
            "max_sharpe": {
                "vol":     round(ms_v * 100, 2),
                "ret":     round(ms_r * 100, 2),
                "sharpe":  round((ms_r - rf) / (ms_v + 1e-9), 3),
                "weights": {t: round(w * 100, 2) for t, w in zip(tickers, ms_weights)},
            },
            "min_vol": {
                "vol":     round(mv_v * 100, 2),
                "ret":     round(mv_r * 100, 2),
                "sharpe":  round((mv_r - rf) / (mv_v + 1e-9), 3),
                "weights": {t: round(w * 100, 2) for t, w in zip(tickers, mv_weights)},
            },
        }
    }


def get_signal_score(strategy_results: list) -> float:
    """
    Calculate composite signal score from strategy results (0~1)
    - Higher Sharpe ratio → higher score
    - Bonus if current position is buy
    """
    scores = []
    for result in strategy_results:
        data = result.get("data", pd.DataFrame())
        metrics = result.get("metrics", {})

        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_score = min(max((sharpe + 1) / 4, 0), 1)  # Normalize -1~3 range to 0~1

        # Current position (last signal)
        if not data.empty and "Signal" in data.columns:
            last_signal = data["Signal"].iloc[-1]
            position_score = 1.0 if last_signal == 1 else 0.3
        else:
            position_score = 0.5

        scores.append((sharpe_score + position_score) / 2)

    return float(np.mean(scores)) if scores else 0.5
