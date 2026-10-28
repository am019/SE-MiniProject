"""Experiment Orchestrator (FR-10, FR-13, FR-14, FR-15, FR-19, NFR-03, NFR-04, NFR-06).

Model services are discovered ONLY from config/models.json; nothing here names a model family.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from fastapi import Body
from pydantic import BaseModel, Field

from ifap_common import client, ctx
from ifap_common.appfactory import create_app, log
from ifap_common.config import load_models_config, service_url
from ifap_common.errors import AppError
from ifap_common.featureset import FeatureSet

app = create_app("orchestrator")
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exp")
_start_lock = threading.Lock()

DISCLAIMER = ("Model output is an experimental statistical estimate, not investment advice. Financial "
              "predictions are uncertain and past performance does not guarantee future results.")
TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT"}


def cfg() -> dict:
    return load_models_config()


def model_specs(c: dict) -> Dict[str, dict]:
    return {m["name"]: m for m in c["models"]}


@app.on_event("startup")
def _recover():
    """Experiments left RUNNING by a previous process can never finish; mark them FAILED."""
    try:
        r = client.call("registry", "GET", "/experiments", params={"status": "QUEUED,RUNNING"}, timeout=5)
        for e in r["experiments"]:
            client.call("registry", "PUT", f"/experiments/{e['experimentId']}",
                        json={"status": "FAILED", "error": "Orchestrator restarted before the run finished."})
    except AppError:
        pass  # registry not up yet; harmless


@app.get("/models/available")
def available():
    c = cfg()
    health = {n: client.health(n, base_url=u).get("status") for n, u in c["services"].items()}
    return {"models": [{**{k: m[k] for k in ("name", "service", "family", "label", "tasks")},
                        "role": m.get("role", "candidate"), "serviceStatus": health.get(m["service"], "UNKNOWN")}
                       for m in c["models"]],
            "baselines": c["baselines"], "defaults": c["defaults"],
            "primaryMetrics": {"VOLATILITY": ["RMSE", "MAE", "QLIKE"], "OPTION_PRICE": ["RMSE", "MAE"]}}


class ExpReq(BaseModel):
    datasetId: str
    task: str
    target: Optional[str] = None
    horizon: int = Field(5, ge=1, le=30)
    models: List[str] = Field(min_length=1)
    primaryMetric: Optional[str] = None
    seed: int = Field(42, ge=0, le=2 ** 31 - 1)
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    windows: Optional[List[int]] = None
    margin: Optional[float] = Field(None, ge=0, le=0.5)
    tieTolerance: Optional[float] = Field(None, ge=0, le=0.2)
    modelConfig: Dict[str, dict] = Field(default_factory=dict)
    timeoutSec: Optional[int] = Field(None, ge=1, le=7200)
    autoPromote: bool = True


@app.post("/experiments", status_code=202)
def start(req: ExpReq):
    c = cfg()
    specs = model_specs(c)
    if req.task not in ("VOLATILITY", "OPTION_PRICE"):
        raise AppError("VALIDATION_FAILED", "task must be VOLATILITY or OPTION_PRICE.", fields=[{"field": "task"}])
    unknown = [m for m in req.models if m not in specs]
    unsupported = [m for m in req.models if m in specs and req.task not in specs[m]["tasks"]]
    if unknown or unsupported:
        raise AppError("VALIDATION_FAILED", "Unknown or unsupported model(s) for this task: "
                       + ", ".join(unknown + unsupported), fields=[{"field": "models"}])
    pm = req.primaryMetric or ("RMSE" if req.task == "VOLATILITY" else "MAE")
    allowed = ["RMSE", "MAE", "QLIKE"] if req.task == "VOLATILITY" else ["RMSE", "MAE"]
    if pm not in allowed:
        raise AppError("VALIDATION_FAILED", f"primaryMetric must be one of {allowed}.",
                       fields=[{"field": "primaryMetric"}])
    ds = client.call("ingestion", "GET", f"/datasets/{req.datasetId}")
    if ds["status"] != "VALID":
        raise AppError("VALIDATION_FAILED", "The dataset has blocking validation issues; no model was started.")
    with _start_lock:
        running = client.call("registry", "GET", "/experiments", params={"status": "QUEUED,RUNNING"})
        if running["experiments"]:
            raise AppError("EXPERIMENT_IN_PROGRESS", "Another experiment is still running.")
        # shared feature set (same for every model) - fails fast with the validation reason
        fbody = {"datasetId": req.datasetId, "task": req.task, "horizon": req.horizon,
                 "dateFrom": req.dateFrom, "dateTo": req.dateTo}
        if req.windows:
            fbody["windows"] = req.windows
        fs = client.call("features", "POST", "/featuresets", json=fbody)
        base = c["baselines"][req.task]
        models = list(dict.fromkeys(req.models))
        auto_base = base not in models
        if auto_base:
            models.insert(0, base)
        admin_sel = client.call("registry", "GET", "/config").get("selection", {})
        defaults = {**c["defaults"], **admin_sel}
        doc = {"status": "QUEUED", "datasetId": req.datasetId, "datasetHash": ds["datasetHash"],
               "task": req.task, "target": fs["targetDefinition"], "horizon": fs["horizon"],
               "requestedModels": req.models, "models": {m: {"status": "QUEUED", "family": specs[m]["family"],
                                                              "service": specs[m]["service"]} for m in models},
               "baseline": base, "baselineAutoAdded": auto_base, "isSample": bool(ds.get("isSample")), "primaryMetric": pm, "seed": req.seed,
               "margin": req.margin if req.margin is not None else defaults["margin"],
               "tieTolerance": req.tieTolerance if req.tieTolerance is not None else defaults["tieTolerance"],
               "dateFrom": req.dateFrom, "dateTo": req.dateTo, "modelConfig": req.modelConfig,
               "timeoutSec": req.timeoutSec or admin_sel.get("timeoutSec"), "autoPromote": req.autoPromote,
               "featureSetId": fs["featureSetId"], "featureSetVersion": fs["featureSetVersion"],
               "testTimestampHash": fs["testTimestampHash"], "splitHash": fs["splitHash"],
               "splitRanges": fs["splitRanges"], "requestId": ctx.request_id.get()}
        exp = client.call("registry", "POST", "/experiments", json=doc)
    _pool.submit(_run, exp["experimentId"], ctx.request_id.get())
    return {"experimentId": exp["experimentId"], "status": "QUEUED", "featureSetId": fs["featureSetId"]}


# ------------------------------------------------------------------- the run
def _run(eid: str, rid: str):
    ctx.request_id.set(rid)  # same correlation ID in every service's logs (NFR-08)
    state_lock = threading.Lock()
    try:
        exp = client.call("registry", "GET", f"/experiments/{eid}")
        client.call("registry", "PUT", f"/experiments/{eid}",
                    json={"status": "RUNNING", "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        c = cfg()
        specs = model_specs(c)
        mstate = exp["models"]
        results: Dict[str, dict] = {}

        def push():
            client.call("registry", "PUT", f"/experiments/{eid}", json={"models": mstate})

        def setstate(name, **kw):
            with state_lock:
                mstate[name].update(kw)
                push()

        def run_model(name: str):
            ctx.request_id.set(rid)
            spec = specs[name]
            base_url = c["services"][spec["service"]]
            timeout = exp.get("timeoutSec") or spec.get("timeoutSec") or c["defaults"]["timeoutSec"]
            mconf = {**spec.get("config", {}), **exp["modelConfig"].get(name, {})}
            setstate(name, status="TRAINING")
            rec = {"name": name, "family": spec["family"], "experimentId": eid, "task": exp["task"],
                   "datasetId": exp["datasetId"], "datasetHash": exp["datasetHash"],
                   "featureSetId": exp["featureSetId"], "featureSetVersion": exp["featureSetVersion"],
                   "seed": exp["seed"], "primaryMetric": exp["primaryMetric"], "isBaseline": name == exp["baseline"],
                   "config": mconf}
            try:
                tr = client.call(spec["service"], "POST", "/train", base_url=base_url, timeout=timeout + 60,
                                 json={"featureSetId": exp["featureSetId"], "model": name, "config": mconf,
                                       "seed": exp["seed"], "timeoutSec": timeout})
                rec.update(modelId=tr["modelId"], status="COMPLETED", config=tr["config"],
                           trainTimeSec=tr["trainTimeSec"], artifactHash=tr["artifactHash"],
                           libraryVersions=tr["libraries"], info=tr.get("info"))
                reg = client.call("registry", "POST", "/models", json=rec)
                setstate(name, status="PREDICTING", modelId=tr["modelId"])
                out = {}
                for split in ("val", "test"):
                    p = client.call(spec["service"], "POST", "/predict", base_url=base_url, timeout=120,
                                    json={"modelId": tr["modelId"], "featureSetId": exp["featureSetId"],
                                          "split": split})
                    ev = client.call("evaluation", "POST", "/evaluate",
                                     json={"featureSetId": exp["featureSetId"], "split": split,
                                           "timestamps": p["timestamps"], "predictions": p["values"]})
                    out[split] = (p, ev)
                rec.update(version=reg["version"], metrics={"val": out["val"][1]["metrics"],
                                                            "test": out["test"][1]["metrics"]},
                           inferenceTimeMs=out["test"][0]["inferenceTimeMs"],
                           testTimestampHash=out["test"][1]["timestampHash"])
                client.call("registry", "PUT", f"/models/{tr['modelId']}", json=rec)
                results[name] = {"model": name, "family": spec["family"], "status": "COMPLETED",
                                 "val": rec["metrics"]["val"], "test": rec["metrics"]["test"],
                                 "inferenceTimeMs": rec["inferenceTimeMs"], "trainTimeSec": rec["trainTimeSec"],
                                 "modelId": tr["modelId"], "testTimestampHash": rec["testTimestampHash"],
                                 "featureSetVersion": exp["featureSetVersion"]}
                setstate(name, status="COMPLETED")
            except AppError as e:
                status = "TIMED_OUT" if e.code == "MODEL_TIMEOUT" else "FAILED"
                reason = {"MODEL_TIMEOUT": "TIMEOUT", "SERVICE_UNAVAILABLE": "SERVICE_UNAVAILABLE"}.get(
                    e.code, e.code) + (": " + e.message if e.code not in ("MODEL_TIMEOUT",) else "")
                log(f"model {name} {status}: {e.code}", logging.WARNING)
                rec.update(status=status, failureReason=reason, modelId=rec.get("modelId") or f"{name}-{eid}-failed")
                # a model that trained but could not be evaluated/predicted is also recorded as failed
                try:
                    client.call("registry", "POST", "/models", json=rec)
                except AppError:
                    pass
                results[name] = {"model": name, "family": spec["family"], "status": status, "reason": reason}
                setstate(name, status=status, reason=reason)

        groups: Dict[str, list] = {}
        for m in mstate:
            groups.setdefault(specs[m]["service"], []).append(m)

        def run_group(ms):
            for m in ms:
                run_model(m)

        threads = [threading.Thread(target=run_group, args=(ms,), daemon=True) for ms in groups.values()]
        [t.start() for t in threads]
        [t.join() for t in threads]

        done = [r for r in results.values() if r["status"] == "COMPLETED"]
        if not done:
            client.call("registry", "PUT", f"/experiments/{eid}",
                        json={"status": "FAILED", "error": "No model completed.", "results": list(results.values()),
                              "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            return
        sel = client.call("evaluation", "POST", "/select", json={
            "task": exp["task"], "primaryMetric": exp["primaryMetric"], "margin": exp["margin"],
            "tieTolerance": exp["tieTolerance"], "baseline": exp["baseline"], "results": list(results.values())})
        champion = sel["selection"]["champion"]
        client.call("registry", "PUT", f"/experiments/{eid}", json={
            "status": "COMPLETED", "leaderboard": sel["leaderboard"], "selection": sel["selection"],
            "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if exp.get("autoPromote", True):  # only after the run completed (FR-19)
            client.call("registry", "POST", "/champion", json={
                "task": exp["task"], "modelId": champion["modelId"], "experimentId": eid,
                "reason": sel["selection"]["justification"],
                "doc": {"primaryMetric": exp["primaryMetric"], "flags": sel["selection"]["flags"]}})
    except Exception as e:  # noqa: BLE001 - never leave an experiment stuck in RUNNING
        logging.getLogger("ifap").error("experiment failed", exc_info=e)
        try:
            client.call("registry", "PUT", f"/experiments/{eid}",
                        json={"status": "FAILED", "error": "The experiment could not be completed."})
        except AppError:
            pass


# --------------------------------------------------------------------- queries
@app.get("/experiments")
def list_exps(limit: int = 50):
    r = client.call("registry", "GET", "/experiments", params={"limit": limit})
    keys = ["experimentId", "status", "task", "datasetId", "horizon", "primaryMetric", "createdAt", "baseline"]
    return {"experiments": [{k: e.get(k) for k in keys} | {"models": list(e.get("models", {}))}
                            for e in r["experiments"]]}


@app.get("/experiments/{eid}")
def get_exp(eid: str):
    e = client.call("registry", "GET", f"/experiments/{eid}")
    return {k: v for k, v in e.items() if k not in ("leaderboard",)}


def _complete(eid: str) -> dict:
    e = client.call("registry", "GET", f"/experiments/{eid}")
    if e["status"] != "COMPLETED":
        raise AppError("EXPERIMENT_NOT_COMPLETE", "The experiment has not completed yet." if e["status"] in
                       ("QUEUED", "RUNNING") else "The experiment did not complete.")
    return e


@app.get("/experiments/{eid}/leaderboard")
def leaderboard(eid: str):
    e = _complete(eid)
    return {"experimentId": eid, "task": e["task"], "primaryMetric": e["primaryMetric"],
            "horizon": e["horizon"], "testPeriod": e["splitRanges"]["test"], "valPeriod": e["splitRanges"]["val"],
            "testTimestampHash": e["testTimestampHash"], "featureSetVersion": e["featureSetVersion"],
            "policy": e["selection"]["policy"], "metricDefinitions": METRIC_DEFS,
            "isSample": bool(e.get("isSample")), "leaderboard": e["leaderboard"]}


@app.get("/experiments/{eid}/selection")
def selection(eid: str):
    e = _complete(eid)
    return e["selection"] | {"experimentId": eid}


METRIC_DEFS = {
    "RMSE": "Root mean squared error, in target units (lower is better).",
    "MAE": "Mean absolute error, in target units (lower is better).",
    "QLIKE": "Quasi-likelihood loss on variance: mean(s2/h - ln(s2/h) - 1); lower is better (volatility only).",
    "MAPE": "Mean absolute percentage error in %; n/a when any actual value is zero or near zero.",
    "DA": "Directional accuracy: share of consecutive test points where the predicted change has the same "
          "sign as the actual change (volatility only; n/a for option prices).",
}


# ----------------------------------------------------------------- predictions
class PredReq(BaseModel):
    task: str = "VOLATILITY"
    horizon: Optional[int] = Field(None, ge=1, le=30)
    strike: Optional[float] = Field(None, gt=0)
    daysToExpiry: Optional[float] = Field(None, gt=0, le=3650)
    optionType: Optional[str] = None
    spot: Optional[float] = Field(None, gt=0)


@app.post("/predictions")
def predict(req: PredReq):
    if req.task not in ("VOLATILITY", "OPTION_PRICE"):
        raise AppError("VALIDATION_FAILED", "task must be VOLATILITY or OPTION_PRICE.", fields=[{"field": "task"}])
    ch = client.call("registry", "GET", "/champion", params={"task": req.task})  # 404 NO_CHAMPION
    m = ch["model"]
    fs = FeatureSet.load(m["featureSetId"])
    if req.task == "VOLATILITY" and req.horizon is not None and req.horizon != fs.horizon:
        raise AppError("VALIDATION_FAILED", f"The champion was trained for a {fs.horizon}-day horizon; request "
                       f"horizon={fs.horizon} or run a new experiment for horizon {req.horizon}.",
                       fields=[{"field": "horizon"}])
    li = client.call("features", "POST", f"/featuresets/{fs.id}/latest-inputs",
                     json={"strike": req.strike, "daysToExpiry": req.daysToExpiry,
                           "optionType": req.optionType, "spot": req.spot})
    spec = model_specs(cfg())[m["name"]]
    p = client.call(spec["service"], "POST", "/predict", base_url=cfg()["services"][spec["service"]], timeout=30,
                    json={"modelId": m["modelId"], "featureSetId": fs.id, "split": "latest",
                          "inputs": li["inputs"]})
    pm = m.get("primaryMetric", "RMSE")
    return {"task": req.task, "model": m["name"], "family": m["family"], "modelId": m["modelId"],
            "version": m.get("version"), "experimentId": m.get("experimentId"), "horizon": fs.horizon or None,
            "asOf": li["asOf"], "unit": fs.meta["unit"], "values": p["values"], "inferenceTimeMs": p["inferenceTimeMs"],
            "validationMetric": {"name": pm, "value": (m.get("metrics", {}).get("val", {}) or {}).get(pm)},
            "datasetId": m["datasetId"], "featureSetVersion": m["featureSetVersion"],
            "simulator": m["family"] == "qml", "isSample": bool(fs.meta.get("isSample")),
            "disclaimer": DISCLAIMER}


# ----------------------------------------------------------------- registry pass-through
@app.get("/champion")
def champion(task: str):
    return client.call("registry", "GET", "/champion", params={"task": task})


@app.get("/champions")
def champions():
    return client.call("registry", "GET", "/champions")


@app.get("/champion/history")
def champion_history(task: Optional[str] = None):
    return client.call("registry", "GET", "/champion/history", params={"task": task} if task else None)


@app.post("/champion/rollback")
def rollback(task: str = Body(..., embed=True)):
    return client.call("registry", "POST", "/champion/rollback", json={"task": task})


@app.get("/models")
def models(experimentId: Optional[str] = None):
    return client.call("registry", "GET", "/models", params={"experimentId": experimentId} if experimentId else None)


@app.get("/models/{mid}")
def model(mid: str):
    return client.call("registry", "GET", f"/models/{mid}")


@app.delete("/models/{mid}")
def delete_model(mid: str):
    return client.call("registry", "DELETE", f"/models/{mid}")


@app.get("/config")
def get_config():
    return client.call("registry", "GET", "/config")


@app.put("/config")
def put_config(values: dict = Body(...)):
    sel = values.get("selection")
    if sel is not None:
        ok = (isinstance(sel, dict) and 0 <= float(sel.get("margin", 0)) <= 0.5
              and 0 <= float(sel.get("tieTolerance", 0.01)) <= 0.2 and 1 <= int(sel.get("timeoutSec", 1200)) <= 7200)
        if not ok:
            raise AppError("CONFIG_INVALID", "selection must have margin 0-0.5, tieTolerance 0-0.2, timeoutSec 1-7200.")
    return client.call("registry", "PUT", "/config", json=values)
