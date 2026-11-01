"""Stub model used to demonstrate NFR-06: it implements the contract, nothing else.

Predicts the longest-window realised volatility (volatility task) or a moneyness-based
intrinsic-value heuristic (option task).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ifap_common.modelservice import BaseModelImpl


class Persistence(BaseModelImpl):
    name, label, tasks = "stub_persistence", "Stub (persistence)", ["VOLATILITY", "OPTION_PRICE"]
    description = "Contract-compliance stub: persistence forecast, no training."
    default_config: dict = {}

    def fit(self, fs, config, seed, out_dir: Path):
        (out_dir / "stub.json").write_text(json.dumps({"seed": seed}))
        return {"seed": seed}

    def predict(self, fs, frame, out_dir: Path):
        raw = frame.X_raw
        if fs.task == "VOLATILITY":
            return fs.col(raw, f"rv_{max(fs.meta['windows'])}")
        S = fs.col(raw, "S")
        K = S * np.exp(-fs.col(raw, "log_moneyness"))
        call = fs.col(raw, "is_call") > 0.5
        return np.where(call, np.maximum(S - K, 0), np.maximum(K - S, 0)) + 0.01 * S
