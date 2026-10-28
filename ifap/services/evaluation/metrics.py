"""Forecast-error metrics (FR-11). Pure numpy; hand-checkable (TC-10)."""
from __future__ import annotations

import numpy as np

VOL_METRICS = ["RMSE", "MAE", "QLIKE", "MAPE", "DA"]
OPT_METRICS = ["RMSE", "MAE", "MAPE", "DA"]
PRIMARY_ALLOWED = {"VOLATILITY": ["RMSE", "MAE", "QLIKE"], "OPTION_PRICE": ["RMSE", "MAE"]}
DEFAULT_PRIMARY = {"VOLATILITY": "RMSE", "OPTION_PRICE": "MAE"}
MAPE_ZERO_TOL = 1e-4  # MAPE is reported as n/a if any actual value is this close to zero


def compute(task: str, y, yhat) -> dict:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape or len(y) == 0:
        raise ValueError("actual and predicted must be non-empty and the same length")
    err = yhat - y
    out = {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err)))}
    if task == "VOLATILITY":
        s2, h = np.maximum(y, 1e-6) ** 2, np.maximum(yhat, 1e-6) ** 2
        out["QLIKE"] = float(np.mean(s2 / h - np.log(s2 / h) - 1.0))
    out["MAPE"] = None if np.any(np.abs(y) < MAPE_ZERO_TOL) else float(np.mean(np.abs(err / y)) * 100.0)
    if task == "VOLATILITY" and len(y) > 1:
        dy, dp = np.diff(y), np.diff(yhat)
        out["DA"] = float(np.mean(np.sign(dy) == np.sign(dp)))
    else:
        out["DA"] = None  # n/a: rows are cross-sectional quotes, not one ordered series
    return out
