"""Feature construction + chronological split (FR-04, FR-05, NFR-05). Pure functions, no I/O."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ifap_common.bs import implied_vol
from ifap_common.errors import AppError
from ifap_common.featureset import FEATURE_CODE_VERSION, ts_hash

TRADING_DAYS = 252
DEFAULT_WINDOWS = [5, 10, 22]


def feature_set_id(dataset_hash: str, cfg: dict) -> str:
    blob = json.dumps({"d": dataset_hash, "c": cfg, "v": FEATURE_CODE_VERSION}, sort_keys=True)
    return "fs-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def realized_vol(r: np.ndarray, w: int) -> np.ndarray:
    """Rolling sample standard deviation of the last w log returns, annualised."""
    s = pd.Series(r).rolling(w, min_periods=w).std(ddof=1).to_numpy()
    return s * np.sqrt(TRADING_DAYS)


def forward_vol(r: np.ndarray, h: int) -> np.ndarray:
    """Target at t: annualised RMS of returns t+1..t+h."""
    sq = pd.Series(np.square(r))
    fwd = sq[::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1)  # mean r^2 over t+1..t+h
    return np.sqrt(fwd.to_numpy() * TRADING_DAYS)


def _split_bounds(n: int, ratios: tuple, embargo: int):
    n_tr = int(np.floor(ratios[0] * n))
    n_va = int(np.floor(ratios[1] * n))
    tr = (0, n_tr)
    va = (n_tr + embargo, n_tr + embargo + n_va)
    te = (va[1] + embargo, n)
    if min(tr[1] - tr[0], va[1] - va[0], te[1] - te[0]) < 10:
        raise AppError("VALIDATION_FAILED", "Not enough data after features and embargo to form train, "
                                            "validation and test sets (need at least 10 rows each).")
    return {"train": tr, "val": va, "test": te}


def _scaler(X_raw: np.ndarray, tr: tuple):
    mean = X_raw[tr[0]:tr[1]].mean(axis=0)  # fitted on TRAIN rows only (FR-05)
    std = X_raw[tr[0]:tr[1]].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def build_volatility(df: pd.DataFrame, horizon: int, windows: list, ratios: tuple):
    px = df["close"].to_numpy(dtype=float)
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    N = len(px)
    r = np.full(N, np.nan)
    r[1:] = np.diff(np.log(px))
    names = [f"rv_{w}" for w in windows] + ["ret", "abs_ret", "ret_lag1", "ret_lag2", "ret_lag3"]
    cols = [realized_vol(r, w) for w in windows]
    rr = pd.Series(r)
    cols += [r, np.abs(r), rr.shift(1).to_numpy(), rr.shift(2).to_numpy(), rr.shift(3).to_numpy()]
    F = np.column_stack(cols)
    y_all = forward_vol(np.nan_to_num(r), horizon)
    first = max(windows) + 3  # rolling windows and lags fully defined
    t_idx = np.arange(N)
    labeled = (t_idx >= first) & (t_idx + horizon <= N - 1)
    pos = t_idx[labeled]
    if len(pos) < 40:
        raise AppError("VALIDATION_FAILED", "Too few rows after windowing for this horizon.")
    X_raw, y, ts = F[pos], y_all[pos], dates[pos]
    sp = _split_bounds(len(pos), ratios, embargo=horizon)
    mean, std = _scaler(X_raw, sp["train"])
    latest_raw = F[N - 1:N]
    arrays = {"X_raw": X_raw, "y": y, "ts": ts.astype("U10"), "pos": pos.astype(np.int64),
              "ret_all": np.nan_to_num(r), "ret_dates": dates.astype("U10"),
              "latest_raw": latest_raw, "latest_ts": np.array([dates[-1]]), "latest_pos": np.array([N - 1]),
              "seq_all": F, "seq_idx": pos.astype(np.int64), "latest_seq_idx": np.array([N - 1])}
    meta = {"featureNames": names, "splits": {k: list(v) for k, v in sp.items()}, "embargoRows": horizon,
            "trainReturnEnd": int(pos[sp["train"][1] - 1]), "targetScaleFeature": None,
            "latestContext": None, "baseline": "garch", "seqNames": names, "staticNames": [],
            "seqFirst": int(first),
            "targetDefinition": f"annualised RMS of daily log returns over the next {horizon} trading days",
            "unit": "annualised volatility (decimal, e.g. 0.20 = 20%)"}
    return arrays, meta, (mean, std), sp


def build_options(df: pd.DataFrame, windows: list, ratios: tuple, default_rate: float):
    d = df.copy()
    d = d[(d["option_price"] > 0)]
    if "rate" in d.columns:
        d["rate"] = d["rate"].fillna(d["rate"].median()).fillna(default_rate)
    else:
        d["rate"] = default_rate
    d["tte"] = (d["expiry"] - d["date"]).dt.days / 365.0
    d = d[d["tte"] > 0].sort_values(["date", "expiry", "strike", "option_type"], kind="mergesort")
    d = d.reset_index(drop=True)
    is_call = (d["option_type"] == "call").to_numpy()
    d["iv"] = implied_vol(d["option_price"].to_numpy(), d["underlying_price"].to_numpy(),
                          d["strike"].to_numpy(), d["tte"].to_numpy(), d["rate"].to_numpy(), is_call)
    # per-date underlying series, realised vol (info available at that date) and previous-day median IV
    daily = d.groupby("date").agg(S=("underlying_price", "median"), iv_med=("iv", "median")).sort_index()
    lr = np.log(daily["S"]).diff().to_numpy()
    for w in windows:
        daily[f"rv_{w}"] = realized_vol(lr, w)
    daily["prev_day_iv"] = daily["iv_med"].shift(1)
    d = d.join(daily[[f"rv_{w}" for w in windows] + ["prev_day_iv"]], on="date")
    d["log_moneyness"] = np.log(d["underlying_price"] / d["strike"])
    d["is_call"] = is_call.astype(float)
    d["S"] = d["underlying_price"]
    names = ["log_moneyness", "tte", "is_call", "prev_day_iv"] + [f"rv_{w}" for w in windows] + ["rate", "S"]
    d["tte"] = d["tte"].astype(float)
    ok = d[names].notna().all(axis=1) & np.isfinite(d[names].to_numpy(dtype=float)).all(axis=1)
    d = d[ok].reset_index(drop=True)
    udates = np.array(sorted(d["date"].unique()))
    if len(udates) < 30:
        raise AppError("VALIDATION_FAILED", "Too few quote dates after windowing.")
    nd = len(udates)
    n_tr, n_va = int(np.floor(ratios[0] * nd)), int(np.floor(ratios[1] * nd))
    d_tr, d_va = udates[n_tr - 1], udates[n_tr + n_va - 1]
    date_arr = d["date"].to_numpy()
    tr_end = int(np.searchsorted(date_arr, d_tr, side="right"))
    va_end = int(np.searchsorted(date_arr, d_va, side="right"))
    sp = {"train": (0, tr_end), "val": (tr_end, va_end), "test": (va_end, len(d))}
    if min(b - a for a, b in sp.values()) < 10:
        raise AppError("VALIDATION_FAILED", "Not enough option rows to form train, validation and test sets.")
    X_raw = d[names].to_numpy(dtype=float)
    y = d["option_price"].to_numpy(dtype=float)
    ts = d["date"].dt.strftime("%Y-%m-%d").to_numpy()
    mean, std = _scaler(X_raw, sp["train"])
    last = daily.iloc[-1]
    ctx = {"date": daily.index[-1].strftime("%Y-%m-%d"), "S": float(last["S"]),
           "prev_day_iv": float(last["iv_med"]) if np.isfinite(last["iv_med"]) else float(np.nanmedian(d["iv"])),
           "rate": float(d["rate"].iloc[-1]), **{f"rv_{w}": float(last[f"rv_{w}"]) for w in windows}}
    default_row = option_row(names, ctx, strike=ctx["S"], days=30, call=True)
    seq_names = ["prev_day_iv"] + [f"rv_{w}" for w in windows]
    seq_all = daily[seq_names].to_numpy(dtype=float)
    seq_ok = np.flatnonzero(np.isfinite(seq_all).all(axis=1))
    day_idx = daily.index.get_indexer(d["date"]).astype(np.int64)
    arrays = {"X_raw": X_raw, "y": y, "ts": ts.astype("U10"),
              "latest_raw": np.array([default_row]), "latest_ts": np.array([ctx["date"]]),
              "seq_all": seq_all, "seq_idx": day_idx, "latest_seq_idx": np.array([len(daily) - 1])}
    meta = {"featureNames": names, "splits": {k: list(v) for k, v in sp.items()}, "embargoRows": 0,
            "targetScaleFeature": "S", "latestContext": ctx, "baseline": "black_scholes",
            "seqNames": seq_names, "staticNames": ["log_moneyness", "tte", "is_call", "rate"],
            "seqFirst": int(seq_ok[0]),
            "impliedVolRows": int(np.isfinite(d["iv"]).sum()),
            "targetDefinition": "observed option price (currency units)",
            "unit": "option price in the underlying's currency"}
    return arrays, meta, (mean, std), sp


def option_row(names: list, ctx: dict, strike: float, days: float, call: bool, spot: float | None = None):
    S = float(spot if spot else ctx["S"])
    vals = {"log_moneyness": float(np.log(S / strike)), "tte": days / 365.0, "is_call": 1.0 if call else 0.0,
            "prev_day_iv": ctx["prev_day_iv"], "rate": ctx["rate"], "S": S}
    for k, v in ctx.items():
        if k.startswith("rv_"):
            vals[k] = v
    return [vals[n] for n in names]


def finalize(arrays: dict, meta: dict, scaler, sp: dict, base_cfg: dict):
    ts = arrays["ts"]
    test_ts = ts[sp["test"][0]:sp["test"][1]]
    val_ts = ts[sp["val"][0]:sp["val"][1]]
    train_ts = ts[sp["train"][0]:sp["train"][1]]
    # FR-05 invariants: strictly chronological, no overlap
    if not (str(train_ts[-1]) <= str(val_ts[0]) and str(val_ts[-1]) <= str(test_ts[0])):
        raise AppError("INTERNAL_ERROR", "Chronological split invariant violated.")
    meta.update(base_cfg)
    meta["scaler"] = {"mean": scaler[0].tolist(), "std": scaler[1].tolist(), "fittedOn": "train"}
    meta["rows"] = int(len(ts))
    meta["splitRanges"] = {k: {"from": str(ts[a]), "to": str(ts[b - 1]), "rows": int(b - a)}
                           for k, (a, b) in sp.items()}
    meta["testTimestampHash"] = ts_hash(test_ts)
    meta["valTimestampHash"] = ts_hash(val_ts)
    meta["splitHash"] = ts_hash([str(sp[k]) for k in ("train", "val", "test")] + list(ts[[0, -1]]))
    return meta
