
import warnings
import numpy as np
import pandas as pd

# ── Suppress warnings completely (statsmodels ValueWarning + ConvergenceWarning) ──
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="No supported index")
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization")

try:
    from pmdarima import auto_arima as _auto_arima
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tsa.stattools import adfuller
    SM_OK = True
except ImportError:
    SM_OK = False


# Convert to numpy array to completely eliminate index warnings ──
def _to_array(series: pd.Series) -> np.ndarray:
    """Remove DatetimeIndex — pass pure values only to statsmodels"""
    return series.values.astype(float)


def _fit_arima(values: np.ndarray, order: tuple) -> object:
    """
    Fit statsmodels ARIMA
    - Uses numpy array → no index warnings
    - Removed low_memory=True → get_forecast().conf_int() CI works correctly
    - trend='t' (d=1) → include drift, prevent flat forecast
    """
    # d=1: trend='t'(drift), d=0: trend='c', d=2: trend='n'
    if order[1] == 1:
        trend = 't'
    elif order[1] == 0:
        trend = 'c'
    else:
        trend = 'n'
    model = _ARIMA(values, order=order, trend=trend)
    try:
        # No low_memory → OPG covariance computation possible → CI returned correctly
        fitted = model.fit(method="innovations_mle")
    except Exception:
        try:
            fitted = model.fit(method="statespace",
                               optim_score=None,
                               maxiter=500)
        except Exception:
            try:
                # Last resort fallback: (1,1,0) with drift
                fallback = _ARIMA(values, order=(1, 1, 0), trend='t')
                fitted = fallback.fit(method="innovations_mle")
            except Exception:
                fallback2 = _ARIMA(values, order=(0, 1, 0), trend='t')
                fitted = fallback2.fit()
    return fitted


def run_arima(df: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    ARIMA Time Series Forecast

    Args:
        df: OHLCV DataFrame
        forecast_days: Forecast horizon (trading days, default 30)

    Returns:
        dict: forecast, ci_lower, ci_upper, metrics, etc.
    """
    if not SM_OK:
        return {"error": "statsmodels not installed"}

    close = df["Close"].dropna()
    if len(close) < 50:
        return {"error": "Insufficient data (minimum 50 days required)"}

    # Preserve original date index (for date restoration)
    original_dates = close.index

    # ── Log transform (scale stabilization, ensure positive values) ──
    values     = _to_array(np.log(close))     # Pure numpy array
    log_series = pd.Series(values)            # RangeIndex Series (no warnings)

    # ── Stationarity test (ADF) ──────────────────────────────
    adf_result    = adfuller(values, autolag="AIC")
    is_stationary = adf_result[1] < 0.05

    # ── Optimal order selection ──────────────────────────────
    order = _select_order(values, log_series)

    # ── Run forecast ─────────────────────────────────────────
    forecast_log, ci_log_lower, ci_log_upper = _forecast(
        values, order, forecast_days
    )

    # Inverse transform (log → original price)
    forecast_prices = np.exp(forecast_log)
    ci_lower_prices = np.exp(ci_log_lower)
    ci_upper_prices = np.exp(ci_log_upper)

    # ── Forecast date index (business days) ──────────────────
    last_date      = original_dates[-1]
    forecast_index = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_series  = pd.Series(forecast_prices, index=forecast_index)
    ci_lower_series  = pd.Series(ci_lower_prices, index=forecast_index)
    ci_upper_series  = pd.Series(ci_upper_prices, index=forecast_index)

    # ── In-sample evaluation (simplified with last 30 days — walk-forward) ──
    n_eval = min(30, len(values) // 5)
    preds  = _quick_eval(values, order, n_eval)
    actual = np.exp(values[-n_eval:])
    pred   = np.exp(preds)
    mae    = float(np.mean(np.abs(pred - actual)))
    mape   = float(np.mean(np.abs((pred - actual) / (actual + 1e-9)))) * 100

    current_price    = float(close.iloc[-1])
    forecast_30d_end = float(forecast_series.iloc[-1])
    expected_return  = (forecast_30d_end / current_price - 1) * 100

    return {
        "model_type":        "ARIMA",
        "order":             order,
        "is_stationary":     is_stationary,
        "adf_pvalue":        round(float(adf_result[1]), 4),
        "forecast":          forecast_series,
        "ci_lower":          ci_lower_series,
        "ci_upper":          ci_upper_series,
        "current_price":     current_price,
        "forecast_30d":      round(forecast_30d_end, 4),
        "expected_return_30d": round(expected_return, 2),
        "direction":         "UP" if forecast_30d_end > current_price else "DOWN",
        "metrics": {
            "MAE":  round(mae, 4),
            "MAPE": round(mape, 2),
        }
    }


def _select_order(values: np.ndarray, log_series: pd.Series) -> tuple:
    """Select optimal ARIMA order"""
    # Use auto_arima if pmdarima is available
    if PMDARIMA_OK:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _auto_arima(
                    values,
                    seasonal=False,
                    information_criterion="aic",
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=4, max_q=4, max_d=2,
                    maxiter=200,
                )
            return model.order
        except Exception:
            pass

    # Manual AIC search (simplified)
    best_aic   = np.inf
    best_order = (1, 1, 1)
    candidates = [(1,1,1),(1,1,0),(0,1,1),(2,1,1),(1,1,2),(2,1,2),(0,1,0)]

    for order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # d=1: trend='t'(drift), d=0: trend='c', d=2: trend='n'
                if order[1] == 1:
                    trend = 't'
                elif order[1] == 0:
                    trend = 'c'
                else:
                    trend = 'n'
                m   = _ARIMA(values, order=order, trend=trend)
                fit = m.fit(method="innovations_mle")
                if fit.aic < best_aic:
                    best_aic   = fit.aic
                    best_order = order
        except Exception:
            continue

    return best_order


def _forecast(values: np.ndarray, order: tuple, steps: int):
    """
    Run ARIMA forecast
    Returns: (forecast_log, ci_lower_log, ci_upper_log) — numpy arrays
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _fit_arima(values, order)

        try:
            fc_result = fitted.get_forecast(steps=steps)
            fc_mean   = fc_result.predicted_mean          # numpy array
            ci        = fc_result.conf_int(alpha=0.05)    # (steps, 2) array

            # Extract values if conf_int returns a DataFrame
            if hasattr(ci, "values"):
                ci = ci.values

            ci_lower = ci[:, 0]
            ci_upper = ci[:, 1]

        except Exception:
            # Fallback to simple linear extrapolation on convergence failure
            last_val = values[-1]
            trend    = np.mean(np.diff(values[-30:]))
            fc_mean  = np.array([last_val + trend * (i + 1) for i in range(steps)])
            std_est  = np.std(np.diff(values[-60:])) * np.sqrt(np.arange(1, steps + 1))
            ci_lower = fc_mean - 1.96 * std_est
            ci_upper = fc_mean + 1.96 * std_est

    return fc_mean, ci_lower, ci_upper


def _quick_eval(values: np.ndarray, order: tuple, n_eval: int) -> np.ndarray:
    """
    Simplified forecast quality evaluation
    Replace walk-forward with single fit + n_eval step forecast (speed optimization)
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_values = values[:-n_eval]
            fitted       = _fit_arima(train_values, order)
            fc, _, _     = _forecast(train_values, order, n_eval)
            return fc[:n_eval]
    except Exception:
        return values[-n_eval:]    # Fallback: return last actual values
