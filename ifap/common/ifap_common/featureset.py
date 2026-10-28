"""Feature-set storage shared by the Feature service, model services and Evaluation.

Layout (inside the shared data volume):  features/<featureSetId>/{data.npz, meta.json}
Arrays are stored with allow_pickle=False (data only, never code).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import subdir
from .errors import AppError

FEATURE_CODE_VERSION = "1"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def ts_hash(ts) -> str:
    return sha256_bytes("\n".join(str(t) for t in ts).encode())


@dataclass
class Frame:
    ts: list
    X_raw: np.ndarray
    X: np.ndarray
    y: Optional[np.ndarray]
    pos: Optional[np.ndarray]  # position in the return series (volatility task)
    seq_idx: Optional[np.ndarray] = None  # index into the sequence table (for sequence models)


class FeatureSet:
    def __init__(self, fs_id: str, meta: dict, arrays):
        self.id = fs_id
        self.meta = meta
        self.a = arrays
        self.names: list = meta["featureNames"]
        self.task: str = meta["task"]
        self.horizon: int = meta["horizon"]
        self.mean = np.array(meta["scaler"]["mean"], dtype=float)
        self.std = np.array(meta["scaler"]["std"], dtype=float)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def path(fs_id: str) -> Path:
        if not fs_id.replace("-", "").replace("_", "").isalnum():
            raise AppError("NOT_FOUND", "Feature set not found.")
        return subdir("features") / fs_id

    @classmethod
    def load(cls, fs_id: str) -> "FeatureSet":
        d = cls.path(fs_id)
        if not (d / "meta.json").exists():
            raise AppError("NOT_FOUND", "Feature set not found.", status=404)
        meta = json.loads((d / "meta.json").read_text())
        arrays = dict(np.load(d / "data.npz", allow_pickle=False))
        return cls(fs_id, meta, arrays)

    # ------------------------------------------------------------------ helpers
    def scale(self, X_raw: np.ndarray) -> np.ndarray:
        return (X_raw - self.mean) / self.std

    def split_range(self, split: str) -> tuple[int, int]:
        r = self.meta["splits"].get(split)
        if r is None:
            raise AppError("VALIDATION_FAILED", f"Unknown split '{split}'.")
        return int(r[0]), int(r[1])

    def frame(self, split: str, inputs: Optional[list] = None) -> Frame:
        if split == "latest":
            ctx = self.a["latest_raw"]
            ts = [str(self.a["latest_ts"][0])] * (len(inputs) if inputs else len(ctx))
            raw = np.array(inputs, dtype=float) if inputs else ctx
            if raw.ndim != 2 or raw.shape[1] != len(self.names):
                raise AppError("VALIDATION_FAILED", "Input vector length does not match the feature set.")
            pos = self.a["latest_pos"] if "latest_pos" in self.a and not inputs else None
            sidx = np.full(len(raw), int(self.a["latest_seq_idx"][0]), dtype=np.int64)
            return Frame(ts, raw, self.scale(raw), None, pos, sidx)
        lo, hi = self.split_range(split)
        raw = self.a["X_raw"][lo:hi]
        y = self.a["y"][lo:hi]
        ts = [str(t) for t in self.a["ts"][lo:hi]]
        pos = self.a["pos"][lo:hi] if "pos" in self.a else None
        return Frame(ts, raw, self.scale(raw), y, pos, self.a["seq_idx"][lo:hi])

    def col(self, raw: np.ndarray, name: str) -> np.ndarray:
        return raw[:, self.names.index(name)]

    @property
    def test_ts_hash(self) -> str:
        return self.meta["testTimestampHash"]

    # ---- sequence support (used by the deep-learning model)
    def seq_tensor(self, frame: "Frame", length: int) -> np.ndarray:
        """(rows, length, n_dynamic) scaled sequences ending at each row's own time step; only past data."""
        cols = [self.names.index(n) for n in self.meta["seqNames"]]
        tab = (np.nan_to_num(self.a["seq_all"]) - self.mean[cols]) / self.std[cols]
        first = int(self.meta["seqFirst"])
        idx = frame.seq_idx[:, None] - np.arange(length - 1, -1, -1)[None, :]
        idx = np.maximum(idx, first)
        return tab[idx]

    def static_matrix(self, frame: "Frame") -> np.ndarray:
        cols = [self.names.index(n) for n in self.meta["staticNames"]]
        return frame.X[:, cols] if cols else np.zeros((len(frame.X), 0))
