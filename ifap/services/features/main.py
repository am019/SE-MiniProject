"""Feature Engineering service (FR-04, FR-05, NFR-05)."""
from __future__ import annotations

import json
import time
from typing import List, Optional

import numpy as np
from fastapi import Body
from pydantic import BaseModel, Field, field_validator

from ifap_common import datasets as D
from ifap_common.appfactory import create_app
from ifap_common.errors import AppError
from ifap_common.featureset import FEATURE_CODE_VERSION, FeatureSet

from . import build as B  # noqa: E402  (package import; run as services.features.main)

app = create_app("features")
SPLIT = (0.6, 0.2, 0.2)


class BuildReq(BaseModel):
    datasetId: str
    task: str
    horizon: int = Field(5, ge=1, le=30)
    windows: List[int] = Field(default_factory=lambda: list(B.DEFAULT_WINDOWS))
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    riskFreeRate: float = 0.02

    @field_validator("windows")
    @classmethod
    def _w(cls, v):
        if not v or any(w < 2 or w > 120 for w in v) or len(set(v)) != len(v) or len(v) > 6:
            raise ValueError("windows must be 1-6 distinct integers between 2 and 120")
        return sorted(v)


@app.post("/featuresets")
def build(req: BuildReq):
    if req.task not in ("VOLATILITY", "OPTION_PRICE"):
        raise AppError("VALIDATION_FAILED", "task must be VOLATILITY or OPTION_PRICE.", fields=[{"field": "task"}])
    meta_ds, df = D.load_clean(req.datasetId)  # blocked/invalid datasets stop here (FR-03)
    want = "PRICE_SERIES" if req.task == "VOLATILITY" else "OPTION_CHAIN"
    if meta_ds["type"] != want:
        raise AppError("VALIDATION_FAILED", f"Task {req.task} requires a {want.replace('_', ' ').lower()} dataset, "
                                            f"but this dataset is a {meta_ds['type'].replace('_', ' ').lower()}.")
    try:
        if req.dateFrom:
            df = df[df["date"] >= np.datetime64(req.dateFrom)]
        if req.dateTo:
            df = df[df["date"] <= np.datetime64(req.dateTo)]
    except ValueError:
        raise AppError("VALIDATION_FAILED", "dateFrom/dateTo must be YYYY-MM-DD.")
    df = df.reset_index(drop=True)
    horizon = req.horizon if req.task == "VOLATILITY" else 0
    cfg = {"task": req.task, "horizon": horizon, "windows": req.windows, "dateFrom": req.dateFrom,
           "dateTo": req.dateTo, "split": list(SPLIT), "riskFreeRate": req.riskFreeRate}
    fs_id = B.feature_set_id(meta_ds["datasetHash"], cfg)
    d = FeatureSet.path(fs_id)
    if (d / "meta.json").exists():  # deterministic id => identical build already stored
        meta = json.loads((d / "meta.json").read_text())
        return _summary(fs_id, meta, reused=True)
    if req.task == "VOLATILITY":
        arrays, meta, scaler, sp = B.build_volatility(df, horizon, req.windows, SPLIT)
    else:
        arrays, meta, scaler, sp = B.build_options(df, req.windows, SPLIT, req.riskFreeRate)
    base = {"featureSetId": fs_id, "featureSetVersion": f"fv{FEATURE_CODE_VERSION}-{fs_id[3:11]}",
            "task": req.task, "horizon": horizon, "windows": req.windows, "config": cfg,
            "datasetId": req.datasetId, "datasetHash": meta_ds["datasetHash"],
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "isSample": bool(meta_ds.get("isSample"))}
    meta = B.finalize(arrays, meta, scaler, sp, base)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "data.npz", **arrays)
    (d / "meta.json").write_text(json.dumps(meta))
    return _summary(fs_id, meta, reused=False)


def _summary(fs_id: str, meta: dict, reused: bool) -> dict:
    keys = ["featureSetId", "featureSetVersion", "task", "horizon", "windows", "datasetId", "datasetHash",
            "featureNames", "rows", "splitRanges", "embargoRows", "testTimestampHash", "valTimestampHash",
            "splitHash", "baseline", "targetDefinition", "unit", "isSample"]
    d = {k: meta.get(k) for k in keys}
    d["scaler"] = {"fittedOn": "train"}
    d["reused"] = reused
    return d


@app.get("/featuresets/{fs_id}")
def get(fs_id: str):
    fs = FeatureSet.load(fs_id)
    return _summary(fs_id, fs.meta, reused=True) | {"scaler": fs.meta["scaler"],
                                                    "splits": fs.meta["splits"],
                                                    "latestContext": fs.meta.get("latestContext")}


@app.get("/featuresets/{fs_id}/data")
def data(fs_id: str, split: str = "test", limit: int = 5):
    """Small preview used by tests (feature vectors, not raw rows of the uploaded file)."""
    fs = FeatureSet.load(fs_id)
    fr = fs.frame(split)
    k = max(1, min(limit, 50))
    return {"featureNames": fs.names, "timestamps": fr.ts[:k], "X_raw": fr.X_raw[:k].tolist(),
            "y": fr.y[:k].tolist()}


class LatestReq(BaseModel):
    strike: Optional[float] = Field(None, gt=0)
    daysToExpiry: Optional[float] = Field(None, gt=0, le=3650)
    optionType: Optional[str] = None
    spot: Optional[float] = Field(None, gt=0)


@app.post("/featuresets/{fs_id}/latest-inputs")
def latest_inputs(fs_id: str, req: LatestReq = Body(...)):
    """Build the model input vector for a prediction request (FR-14)."""
    fs = FeatureSet.load(fs_id)
    if fs.task == "VOLATILITY":
        return {"inputs": None, "asOf": str(fs.a["latest_ts"][0])}
    if req.strike is None or req.daysToExpiry is None or req.optionType not in ("call", "put"):
        raise AppError("VALIDATION_FAILED",
                       "Option price prediction needs strike, daysToExpiry and optionType (call|put).",
                       fields=[{"field": f} for f in ("strike", "daysToExpiry", "optionType")])
    row = B.option_row(fs.names, fs.meta["latestContext"], req.strike, req.daysToExpiry,
                       req.optionType == "call", req.spot)
    return {"inputs": [row], "asOf": fs.meta["latestContext"]["date"]}
