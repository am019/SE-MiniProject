"""Evaluation & Model Selection service (FR-11, FR-12, FR-13, NFR-11)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from ifap_common.appfactory import create_app
from ifap_common.errors import AppError
from ifap_common.featureset import FeatureSet, ts_hash

from . import metrics as M

app = create_app("evaluation")


class EvalReq(BaseModel):
    featureSetId: str
    split: str
    timestamps: List[str]
    predictions: List[float]


@app.post("/evaluate")
def evaluate(req: EvalReq):
    if req.split not in ("val", "test"):
        raise AppError("VALIDATION_FAILED", "split must be val or test.")
    fs = FeatureSet.load(req.featureSetId)
    fr = fs.frame(req.split)
    if len(req.predictions) != len(fr.y) or ts_hash(req.timestamps) != ts_hash(fr.ts):
        raise AppError("VALIDATION_FAILED", "Predictions do not cover exactly the feature set's "
                                            f"{req.split} timestamps; comparison would not be like-for-like.")
    if not np.all(np.isfinite(req.predictions)):
        raise AppError("VALIDATION_FAILED", "Predictions contain non-finite values.")
    return {"split": req.split, "n": len(fr.y), "metrics": M.compute(fs.task, fr.y, req.predictions),
            "timestampHash": ts_hash(fr.ts), "featureSetVersion": fs.meta["featureSetVersion"]}


class MetricsReq(BaseModel):
    task: str
    actual: List[float]
    predicted: List[float]


@app.post("/metrics")
def metrics(req: MetricsReq):
    try:
        return {"metrics": M.compute(req.task, req.actual, req.predicted)}
    except ValueError as e:
        raise AppError("VALIDATION_FAILED", str(e))


# ------------------------------------------------------------------ selection
class ModelResult(BaseModel):
    model: str
    family: str = ""
    status: str  # COMPLETED | FAILED | TIMED_OUT
    reason: Optional[str] = None
    val: dict = Field(default_factory=dict)
    test: dict = Field(default_factory=dict)
    inferenceTimeMs: Optional[float] = None
    trainTimeSec: Optional[float] = None
    modelId: Optional[str] = None
    testTimestampHash: Optional[str] = None
    featureSetVersion: Optional[str] = None


class SelectReq(BaseModel):
    task: str = "VOLATILITY"
    primaryMetric: Optional[str] = None
    margin: float = Field(0.0, ge=0.0, le=0.5)        # fraction, 0.02 = must beat baseline by 2%
    tieTolerance: float = Field(0.01, ge=0.0, le=0.2)  # 1% by default (policy)
    baseline: str
    results: List[ModelResult]


def select(req: SelectReq) -> dict:
    """The fixed selection policy (SRS section 3.5). Uses VALIDATION metrics only."""
    pm = req.primaryMetric or M.DEFAULT_PRIMARY.get(req.task, "RMSE")
    if pm not in M.PRIMARY_ALLOWED.get(req.task, ["RMSE", "MAE"]):
        raise AppError("VALIDATION_FAILED", f"primaryMetric {pm} is not supported for {req.task}.")
    done = [r for r in req.results if r.status == "COMPLETED" and r.val.get(pm) is not None]
    base = next((r for r in done if r.model == req.baseline), None)
    excluded = [{"model": r.model, "reason": r.reason or r.status, "status": r.status}
                for r in req.results if r not in done]
    flags, eligible = [], []
    if base is None:
        flags.append("BASELINE_UNAVAILABLE")
        eligible = list(done)
    else:
        thr = base.val[pm] * (1.0 - req.margin)
        eligible = [r for r in done if r.model != req.baseline and r.val[pm] < thr]
    why = {}
    for r in done:
        if r.model == req.baseline:
            why[r.model] = "baseline (reference)"
        elif base is not None and r not in eligible:
            why[r.model] = f"did not beat baseline on validation {pm}"
        else:
            why[r.model] = "eligible"
    champion = None
    tie_note = ""
    if eligible:
        best = min(r.val[pm] for r in eligible)
        tied = [r for r in eligible if r.val[pm] <= best * (1.0 + req.tieTolerance)]
        if len(tied) > 1:
            champion = min(tied, key=lambda r: (r.inferenceTimeMs if r.inferenceTimeMs is not None else 1e18,
                                                r.val[pm]))
            tie_note = (f" {len(tied)} models were within {req.tieTolerance:.0%} of the best validation {pm}; "
                        f"the tie was broken by lower inference time.")
        else:
            champion = tied[0]
    elif base is not None:
        champion = base
        flags.append("NO_MODEL_BEAT_BASELINE")
    elif done:
        champion = min(done, key=lambda r: r.val[pm])
    else:
        raise AppError("VALIDATION_FAILED", "No model completed; nothing can be selected.", status=409)

    ranked = sorted(done, key=lambda r: r.val[pm])
    rows, rank = [], 0
    for r in ranked:
        rank += 1
        rows.append({**r.model_dump(), "rank": rank, "isChampion": r.model == champion.model,
                     "isBaseline": r.model == req.baseline, "eligible": r in eligible,
                     "note": why.get(r.model)})
    for r in req.results:
        if r not in done:
            rows.append({**r.model_dump(), "rank": None, "isChampion": False,
                         "isBaseline": r.model == req.baseline, "eligible": False,
                         "note": f"excluded: {r.reason or r.status}"})
    hashes = {r.testTimestampHash for r in done if r.testTimestampHash}
    versions = {r.featureSetVersion for r in done if r.featureSetVersion}
    integrity = {"identicalTestTimestampHash": len(hashes) <= 1, "identicalFeatureSetVersion": len(versions) <= 1,
                 "testTimestampHash": next(iter(hashes), None), "featureSetVersion": next(iter(versions), None)}
    if "NO_MODEL_BEAT_BASELINE" in flags:
        just = (f"No model beat the baseline ({req.baseline}) on validation {pm} by the required margin "
                f"({req.margin:.0%}); the baseline was selected and the experiment is flagged "
                f"'no model beat the baseline'.")
    elif "BASELINE_UNAVAILABLE" in flags:
        just = (f"The baseline ({req.baseline}) did not complete, so the comparison reference is missing; "
                f"{champion.model} has the lowest validation {pm} among completed models.")
    else:
        just = (f"{champion.model} ({champion.family}) has the lowest validation {pm} "
                f"({champion.val[pm]:.6g}) among {len(eligible)} eligible model(s), versus the baseline "
                f"{req.baseline} ({base.val[pm]:.6g}).{tie_note} Test-period metrics were reported but did not "
                f"influence the choice.")
    if excluded:
        just += " Excluded: " + "; ".join(f"{e['model']} ({e['reason']})" for e in excluded) + "."
    sel = {"champion": {"model": champion.model, "family": champion.family, "modelId": champion.modelId},
           "primaryMetric": pm, "validationValue": champion.val[pm], "baseline": req.baseline,
           "flags": flags, "excluded": excluded, "justification": just,
           "policy": {"margin": req.margin, "tieTolerance": req.tieTolerance, "primaryMetric": pm,
                      "selectionData": "validation only"}, "integrity": integrity}
    return {"leaderboard": rows, "selection": sel}


@app.post("/select")
def select_endpoint(req: SelectReq):
    return select(req)
