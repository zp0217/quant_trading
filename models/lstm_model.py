
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

try:
    import absl.logging as _absl_log
    _absl_log.set_verbosity(_absl_log.ERROR)
    _absl_log._warn_preinit_stderr = False
except Exception:
    pass
for _n in ("tensorflow","tensorflow_core","tensorflow.python","absl","keras"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

FEATURES   = ["Close", "Volume", "RSI", "MACD", "BB_PctB"]
FORECAST_D = 30
N_MC       = 10
MIN_ROWS   = 30   # ~22 trading days (1 month) + buffer


def _adaptive_lookback(n: int) -> int:
    return max(3, min(20, n // 4))


def _make_seqs(arr, lb):
    X, y = [], []
    for i in range(lb, len(arr)):
        X.append(arr[i-lb:i])
        y.append(arr[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _safe_predict(model, X, batch_size, training=False):
    out = []
    for i in range(0, len(X), batch_size):
        chunk = X[i:i+batch_size].astype(np.float32)
        if len(chunk) < batch_size:
            pad   = np.zeros((batch_size - len(chunk), *chunk.shape[1:]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
            out.append(model(chunk, training=training).numpy()[:batch_size - len(pad)])
        else:
            out.append(model(chunk, training=training).numpy())
    return np.concatenate(out, axis=0).flatten()


def _inv_close(arr, scaler):
    dummy = np.zeros((len(arr), len(FEATURES)), dtype=np.float32)
    dummy[:, 0] = arr
    return scaler.inverse_transform(dummy)[:, 0]


def _forecast_future(model, full_sc, scaler, lookback, forecast_days, batch_size, n_mc=N_MC):
    seed = full_sc[-lookback:].copy().astype(np.float32)
    all_runs = []
    for _ in range(n_mc):
        seq, preds = seed.copy(), []
        for _ in range(forecast_days):
            x_in      = seq.reshape(1, lookback, len(FEATURES)).astype(np.float32)
            p         = float(model(x_in, training=True).numpy()[0, 0])
            preds.append(p)
            nxt       = seq[-1].copy()
            nxt[0]    = p
            nxt[1]    = float(np.mean(seq[-5:, 1]))
            delta     = p - seq[-1, 0]
            nxt[2]    = float(np.clip(seq[-1, 2] + delta * 20.0, 0.0, 1.0))
            nxt[3]    = float(seq[-1, 3] + delta * 0.5)
            nxt[4]    = float(np.clip(seq[-1, 4] + delta * 10.0, 0.0, 1.0))
            seq       = np.vstack([seq[1:], nxt])
        all_runs.append(preds)
    mc   = np.array(all_runs)
    mean = mc.mean(axis=0)
    std  = mc.std(axis=0)
    return (_inv_close(mean, scaler),
            _inv_close(mean - 1.96*std, scaler),
            _inv_close(mean + 1.96*std, scaler))


def train_lstm(df: pd.DataFrame, lookback: int = None, epochs: int = 50,
               batch_size: int = 32, forecast_days: int = FORECAST_D) -> dict:
    if not TF_AVAILABLE:
        return {"error": "TensorFlow not available"}

    data = df[FEATURES].dropna().copy()
    n    = len(data)

    if n < MIN_ROWS:
        return {"error": f"LSTM: {n} rows of data (minimum {MIN_ROWS} rows required)",
                "insufficient_data": True}

    lb = lookback if lookback else _adaptive_lookback(n)

    # Split: increase train ratio for smaller n
    if n < 60:
        train_end = max(lb + 5, int(n * 0.80))
        val_end   = max(train_end + 3, int(n * 0.90))
    else:
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

    train_end = max(train_end, lb + 5)
    val_end   = max(val_end, train_end + 3)
    if val_end >= n - lb:
        val_end = max(train_end + 2, n - lb - 1)
    if train_end >= val_end:
        train_end = val_end - 2
    if train_end < lb + 2:
        return {"error": f"LSTM: Insufficient data after split (n={n}, lb={lb})",
                "insufficient_data": True}

    scaler   = MinMaxScaler()
    train_sc = scaler.fit_transform(data.iloc[:train_end].values.astype(np.float32))
    val_sc   = scaler.transform(data.iloc[train_end:val_end].values.astype(np.float32))
    test_sc  = scaler.transform(data.iloc[val_end:].values.astype(np.float32))
    full_sc  = np.vstack([train_sc, val_sc, test_sc])

    X_train, y_train = _make_seqs(train_sc, lb)
    X_val,   y_val   = _make_seqs(np.vstack([train_sc[-lb:], val_sc]), lb)
    X_test,  y_test  = _make_seqs(full_sc[-(len(test_sc) + lb):], lb) if len(test_sc) > lb else (np.array([]), np.array([]))

    if len(X_train) < 3:
        return {"error": f"LSTM: Insufficient train sequences ({len(X_train)})",
                "insufficient_data": True}

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lb, len(FEATURES))),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val) if len(X_val) > 0 else None,
              epochs=min(epochs, 30 if n < 60 else epochs),
              batch_size=max(1, min(batch_size, len(X_train))),
              callbacks=[
                  EarlyStopping(monitor="val_loss" if len(X_val)>0 else "loss",
                                patience=6, restore_best_weights=True),
              ], verbose=0)

    last_seq = full_sc[-lb:].reshape(1, lb, len(FEATURES)).astype(np.float32)
    if len(X_test) > 0:
        X_all    = np.concatenate([X_test, last_seq], axis=0)
        all_pred = _safe_predict(model, X_all, max(1, min(batch_size, len(X_all))), training=False)
        pred_prices   = _inv_close(all_pred[:-1], scaler)
        actual_prices = _inv_close(y_test, scaler)
        next_price    = _inv_close(all_pred[-1:], scaler)[0]
    else:
        pred_prices   = np.array([])
        actual_prices = np.array([])
        next_price    = _inv_close(_safe_predict(model, last_seq, 1, training=False), scaler)[0]

    mape = (float(np.mean(np.abs((pred_prices - actual_prices) / (actual_prices + 1e-9)))) * 100
            if len(pred_prices) > 0 else 0.0)
    mse  = float(np.mean((pred_prices - actual_prices)**2)) if len(pred_prices) > 0 else 0.0

    fc_mean, ci_lower, ci_upper = _forecast_future(
        model, full_sc, scaler, lb, forecast_days, max(1, min(batch_size, 8)))
    last_date = data.index[-1]
    fc_index  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    current   = float(data["Close"].iloc[-1])
    ret30     = (float(fc_mean[-1]) - current) / current * 100

    return {
        "model_type":    "LSTM",
        "lookback_used": lb,
        "pred_prices":   pred_prices,
        "actual_prices": actual_prices,
        "test_index":    data.index[-len(pred_prices):] if len(pred_prices) > 0 else pd.DatetimeIndex([]),
        "fc_mean":       fc_mean,
        "ci_lower":      ci_lower,
        "ci_upper":      ci_upper,
        "fc_index":      fc_index,
        "next_day_pred": round(float(next_price), 4),
        "forecast_30d":  round(float(fc_mean[-1]), 4),
        "ret30":         round(ret30, 2),
        "current_price": current,
        "direction":     "UP" if fc_mean[-1] > current else "DOWN",
        "metrics":       {"MSE": round(mse,4), "MAE": 0.0, "MAPE": round(mape,2)},
    }
