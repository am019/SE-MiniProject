"""Common model-service contract (FR-09 / NFR-06 / SW-01).

Every model family service (classical, dl, qml, stub, ...) is created with
`create_model_app(...)` and therefore exposes the identical API:

    POST /train     {featureSetId, model, config, seed, timeoutSec}
    POST /predict   {modelId, featureSetId, split | inputs}
    GET  /describe
    GET  /health

Training runs in an isolated process with time/memory limits (SR-07, NFR-03).
Artifacts are hashed (SHA-256); the expected hash is read from the Registry and
verified before every load (SR-05).
"""
from __future__ import annotations

import importlib
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import Request
from pydantic import BaseModel, Field

from . import client
from .appfactory import create_app
from .config import env, env_int, subdir
from .errors import AppError
from .featureset import FeatureSet, sha256_dir
from .isolated import run_isolated


class BaseModelImpl:
    """Interface implemented by every concrete model."""
    name = "base"
    label = "Base"
    tasks = ["VOLATILITY"]
    default_config: dict = {}
    description = ""

    bounds: dict = {}  # {key: (min, max)} numeric bounds enforced by validate_config

    def validate_config(self, config: dict) -> dict:
        merged = dict(self.default_config)
        merged.update({k: v for k, v in (config or {}).items() if k != "fault"})
        for k, v in merged.items():
            if k not in self.default_config:
                raise AppError("CONFIG_INVALID", f"Unknown parameter '{k}'.", fields=[{"field": k}])
            if k in self.bounds and v is not None:
                lo, hi = self.bounds[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
                    raise AppError("CONFIG_INVALID", f"Parameter '{k}' must be a number between {lo} and {hi}.",
                                   fields=[{"field": k}])
        return merged

    @staticmethod
    def target_scale(fs, raw: np.ndarray) -> np.ndarray:
        """Per-row divisor for the target (spot for option prices, 1 for volatility)."""
        n = fs.meta.get("targetScaleFeature")
        return raw[:, fs.names.index(n)] if n else np.ones(len(raw))

    def fit(self, fs: FeatureSet, config: dict, seed: int, out_dir: Path) -> dict:  # pragma: no cover
        raise NotImplementedError

    def predict(self, fs: FeatureSet, frame, out_dir: Path) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:
        return {"name": self.name, "label": self.label, "tasks": self.tasks,
                "defaultConfig": self.default_config, "description": self.description}


# ----------------------------------------------------------------- child job
def _load_impl(spec: str) -> BaseModelImpl:
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _train_job(spec: str, fs_id: str, config: dict, seed: int, out_dir: str, fault: Optional[str]):
    """Runs inside the isolated child process."""
    if fault and os.environ.get("IFAP_ALLOW_FAULT_INJECTION") == "1":
        if fault == "crash":
            raise RuntimeError("injected fault")
        if fault == "hang":
            time.sleep(10 ** 6)
        if fault == "oom":
            _hog = bytearray(8 * 1024 ** 3)  # noqa: F841 - exceeds the memory limit
    impl = _load_impl(spec)
    fs = FeatureSet.load(fs_id)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    info = impl.fit(fs, config, seed, Path(out_dir))
    return info or {}


# --------------------------------------------------------------------- schemas
class TrainReq(BaseModel):
    featureSetId: str
    model: str
    config: dict = Field(default_factory=dict)
    seed: int = 42
    timeoutSec: int = 1200


class PredictReq(BaseModel):
    modelId: str
    featureSetId: str
    split: str = "latest"
    inputs: Optional[list] = None
    horizon: Optional[int] = None


def lib_versions(extra: list[str] | None = None) -> dict:
    import importlib.metadata as md
    out = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "arch", "torch", "pennylane"] + (extra or []):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def create_model_app(service: str, family: str, models: dict[str, str], extra_libs=None):
    """models: {modelName: 'python.module:ClassName'}"""
    app = create_app(service)
    mem_mb = env_int("IFAP_JOB_MEM_MB", 6144)
    registry = lambda: client.call("registry", "GET", "/models/{}")  # noqa: E731 (kept for clarity)
    impls = {n: _load_impl(s) for n, s in models.items()}
    models_root = subdir("models")

    def artifact_dir(model_id: str) -> Path:
        if not model_id.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise AppError("MODEL_NOT_FOUND", "Model not found.")
        return models_root / model_id

    @app.get("/describe")
    def describe():
        return {"service": service, "family": family, "contract": "1.0",
                "maxQubits": env_int("IFAP_MAX_QUBITS", 8) if family == "qml" else None,
                "models": [i.info() for i in impls.values()], "libraries": lib_versions(extra_libs),
                "limits": {"memoryMB": mem_mb, "defaultTimeoutSec": 1200}}

    @app.post("/train")
    def train(req: TrainReq):
        impl = impls.get(req.model)
        if impl is None:
            raise AppError("CONFIG_INVALID", f"Unknown model '{req.model}' for this service.")
        fs = FeatureSet.load(req.featureSetId)
        if fs.task not in impl.tasks:
            raise AppError("CONFIG_INVALID", f"Model '{req.model}' does not support task {fs.task}.")
        config = impl.validate_config(req.config)  # raises CONFIG_INVALID / QUBIT_LIMIT_EXCEEDED
        model_id = f"{req.model}-{uuid.uuid4().hex[:8]}"
        out = artifact_dir(model_id)
        t0 = time.time()
        try:
            info = run_isolated(_train_job, (models[req.model], req.featureSetId, config, req.seed,
                                             str(out), (req.config or {}).get("fault")),
                                timeout_s=req.timeoutSec, mem_mb=mem_mb)
        except AppError:
            shutil.rmtree(out, ignore_errors=True)
            raise
        train_time = round(time.time() - t0, 3)
        return {"modelId": model_id, "model": req.model, "family": family, "status": "COMPLETED",
                "trainTimeSec": train_time, "config": config, "seed": req.seed,
                "artifactHash": sha256_dir(out), "info": info, "libraries": lib_versions(extra_libs)}

    @app.post("/predict")
    def predict(req: PredictReq):
        model_name = req.modelId.rsplit("-", 1)[0]
        impl = impls.get(model_name)
        out = artifact_dir(req.modelId)
        if impl is None or not out.exists():
            raise AppError("MODEL_NOT_FOUND", "Model not found.")
        # SR-05: verify artifact integrity against the registry hash before loading
        rec = client.call("registry", "GET", f"/models/{req.modelId}")
        if sha256_dir(out) != rec.get("artifactHash"):
            logging.getLogger("ifap").error("artifact hash mismatch for %s", req.modelId)
            raise AppError("ARTIFACT_INTEGRITY_FAILED",
                           "Model artifact failed its integrity check and was not loaded.")
        fs = FeatureSet.load(req.featureSetId)
        if req.split not in ("val", "test", "latest"):
            raise AppError("VALIDATION_FAILED", "split must be val, test or latest.")
        frame = fs.frame(req.split, req.inputs)
        t0 = time.perf_counter()
        values = np.asarray(impl.predict(fs, frame, out), dtype=float)
        ms = (time.perf_counter() - t0) * 1000.0
        if not np.all(np.isfinite(values)):
            raise AppError("MODEL_FAILED", "Model produced non-finite predictions.")
        return {"modelId": req.modelId, "timestamps": frame.ts, "values": [float(v) for v in values],
                "inferenceTimeMs": round(ms, 3), "perRowMicros": round(ms * 1000 / max(len(values), 1), 3)}

    return app
