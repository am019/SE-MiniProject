"""Model Registry & Artifact Store service (FR-16, FR-19, SR-05).

SQLite (WAL) persistence for model runs, experiments, champion history and
platform configuration. Artifact bytes live on the shared volume; the registry
stores their SHA-256 hash.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Optional

from fastapi import Body
from pydantic import BaseModel

from ifap_common.appfactory import create_app
from ifap_common.config import data_dir, subdir
from ifap_common.errors import AppError

app = create_app("registry")
DB = data_dir() / "registry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  modelId TEXT PRIMARY KEY, experimentId TEXT, name TEXT, family TEXT, version TEXT,
  task TEXT, status TEXT, createdAt TEXT, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
  experimentId TEXT PRIMARY KEY, seq INTEGER UNIQUE, status TEXT, createdAt TEXT, updatedAt TEXT,
  doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS champions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, modelId TEXT, experimentId TEXT,
  promotedAt TEXT, retiredAt TEXT, active INTEGER, reason TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


with conn() as _c:
    _c.executescript(SCHEMA)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ models
@app.post("/models")
def register_model(doc: dict = Body(...)):
    mid = doc.get("modelId")
    if not mid or not doc.get("name"):
        raise AppError("VALIDATION_FAILED", "modelId and name are required.")
    with conn() as c:
        n = c.execute("SELECT COUNT(*) FROM models WHERE name=?", (doc["name"],)).fetchone()[0]
        doc.setdefault("version", f"v{n + 1}")
        doc.setdefault("createdAt", now())
        c.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?,?,?)",
                  (mid, doc.get("experimentId"), doc["name"], doc.get("family"), doc["version"],
                   doc.get("task"), doc.get("status", "COMPLETED"), doc["createdAt"], json.dumps(doc)))
    return _with_champion_flag(doc)


def _champion_ids() -> set:
    with conn() as c:
        return {r["modelId"] for r in c.execute("SELECT modelId FROM champions WHERE active=1")}


def _with_champion_flag(doc: dict) -> dict:
    doc = dict(doc)
    doc["isChampion"] = doc.get("modelId") in _champion_ids()
    return doc


@app.put("/models/{model_id}")
def update_model(model_id: str, patch: dict = Body(...)):
    with conn() as c:
        r = c.execute("SELECT doc FROM models WHERE modelId=?", (model_id,)).fetchone()
        if not r:
            raise AppError("MODEL_NOT_FOUND", "Model not found.")
        doc = json.loads(r["doc"])
        doc.update(patch)
        c.execute("UPDATE models SET status=?, doc=? WHERE modelId=?",
                  (doc.get("status"), json.dumps(doc), model_id))
    return _with_champion_flag(doc)


@app.get("/models/{model_id}")
def get_model(model_id: str):
    with conn() as c:
        r = c.execute("SELECT doc FROM models WHERE modelId=?", (model_id,)).fetchone()
    if not r:
        raise AppError("MODEL_NOT_FOUND", "Model not found.")
    return _with_champion_flag(json.loads(r["doc"]))


@app.get("/models")
def list_models(experimentId: Optional[str] = None, task: Optional[str] = None):
    q, args = "SELECT doc FROM models", []
    conds = []
    if experimentId:
        conds.append("experimentId=?"); args.append(experimentId)
    if task:
        conds.append("task=?"); args.append(task)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY createdAt, modelId"
    with conn() as c:
        rows = [json.loads(r["doc"]) for r in c.execute(q, args)]
    ch = _champion_ids()
    for d in rows:
        d["isChampion"] = d["modelId"] in ch
    return {"models": rows}


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if model_id in _champion_ids():
        raise AppError("VALIDATION_FAILED", "The active champion cannot be deleted; roll back first.", status=409)
    with conn() as c:
        r = c.execute("SELECT 1 FROM models WHERE modelId=?", (model_id,)).fetchone()
        if not r:
            raise AppError("MODEL_NOT_FOUND", "Model not found.")
        c.execute("DELETE FROM models WHERE modelId=?", (model_id,))
    shutil.rmtree(subdir("models") / model_id, ignore_errors=True)
    return {"deleted": model_id}


# ------------------------------------------------------------- experiments
@app.post("/experiments")
def create_experiment(doc: dict = Body(...)):
    with conn() as c:
        seq = (c.execute("SELECT COALESCE(MAX(seq),0) FROM experiments").fetchone()[0]) + 1
        eid = f"exp-{seq:04d}"
        doc.update({"experimentId": eid, "createdAt": now(), "updatedAt": now()})
        doc.setdefault("status", "QUEUED")
        c.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                  (eid, seq, doc["status"], doc["createdAt"], doc["updatedAt"], json.dumps(doc)))
    return doc


@app.put("/experiments/{eid}")
def update_experiment(eid: str, patch: dict = Body(...)):
    with conn() as c:
        r = c.execute("SELECT doc FROM experiments WHERE experimentId=?", (eid,)).fetchone()
        if not r:
            raise AppError("EXPERIMENT_NOT_FOUND", "Experiment not found.")
        doc = json.loads(r["doc"])
        doc.update(patch)
        doc["updatedAt"] = now()
        c.execute("UPDATE experiments SET status=?, updatedAt=?, doc=? WHERE experimentId=?",
                  (doc.get("status"), doc["updatedAt"], json.dumps(doc), eid))
    return doc


@app.get("/experiments/{eid}")
def get_experiment(eid: str):
    with conn() as c:
        r = c.execute("SELECT doc FROM experiments WHERE experimentId=?", (eid,)).fetchone()
    if not r:
        raise AppError("EXPERIMENT_NOT_FOUND", "Experiment not found.")
    return json.loads(r["doc"])


@app.get("/experiments")
def list_experiments(status: Optional[str] = None, limit: int = 100):
    q, args = "SELECT doc FROM experiments", []
    if status:
        q += " WHERE status IN (%s)" % ",".join("?" * len(status.split(",")))
        args = status.split(",")
    q += " ORDER BY seq DESC LIMIT ?"
    args.append(limit)
    with conn() as c:
        return {"experiments": [json.loads(r["doc"]) for r in c.execute(q, args)]}


# --------------------------------------------------------------- champions
class PromoteReq(BaseModel):
    task: str
    modelId: str
    experimentId: Optional[str] = None
    reason: str = ""
    doc: dict = {}


@app.post("/champion")
def promote(req: PromoteReq):
    """Promote only an existing, COMPLETED model; the previous champion is retained (FR-19)."""
    with conn() as c:
        m = c.execute("SELECT status FROM models WHERE modelId=?", (req.modelId,)).fetchone()
        if not m:
            raise AppError("MODEL_NOT_FOUND", "Model not found.")
        if m["status"] != "COMPLETED":
            raise AppError("VALIDATION_FAILED", "Only COMPLETED models can be promoted.", status=409)
        c.execute("UPDATE champions SET active=0, retiredAt=? WHERE task=? AND active=1", (now(), req.task))
        c.execute("INSERT INTO champions (task,modelId,experimentId,promotedAt,active,reason,doc) "
                  "VALUES (?,?,?,?,1,?,?)",
                  (req.task, req.modelId, req.experimentId, now(), req.reason, json.dumps(req.doc)))
    return get_champion(req.task)


def _champ_row(r) -> dict:
    return {"task": r["task"], "modelId": r["modelId"], "experimentId": r["experimentId"],
            "promotedAt": r["promotedAt"], "retiredAt": r["retiredAt"], "active": bool(r["active"]),
            "reason": r["reason"], "doc": json.loads(r["doc"] or "{}")}


@app.get("/champion")
def get_champion(task: str):
    with conn() as c:
        r = c.execute("SELECT * FROM champions WHERE task=? AND active=1", (task,)).fetchone()
    if not r:
        raise AppError("NO_CHAMPION", "No champion model exists for this task yet.")
    ch = _champ_row(r)
    ch["model"] = get_model(ch["modelId"])
    return ch


@app.get("/champions")
def all_champions():
    out = {}
    with conn() as c:
        for r in c.execute("SELECT * FROM champions WHERE active=1"):
            out[r["task"]] = _champ_row(r)
    return out


@app.get("/champion/history")
def champion_history(task: Optional[str] = None):
    with conn() as c:
        if task:
            rows = c.execute("SELECT * FROM champions WHERE task=? ORDER BY id DESC", (task,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM champions ORDER BY id DESC").fetchall()
    return {"history": [_champ_row(r) for r in rows]}


@app.post("/champion/rollback")
def rollback(task: str = Body(..., embed=True)):
    with conn() as c:
        cur = c.execute("SELECT * FROM champions WHERE task=? AND active=1", (task,)).fetchone()
        prev = c.execute("SELECT * FROM champions WHERE task=? AND active=0 ORDER BY id DESC",
                         (task,)).fetchall()
        # restore the most recent previous champion whose model still exists
        target = None
        for p in prev:
            if p["modelId"] != (cur["modelId"] if cur else None) and c.execute(
                    "SELECT 1 FROM models WHERE modelId=?", (p["modelId"],)).fetchone():
                target = p
                break
        if not target:
            raise AppError("NO_PREVIOUS_CHAMPION", "There is no previous champion to restore.")
        c.execute("UPDATE champions SET active=0, retiredAt=? WHERE task=? AND active=1", (now(), task))
        c.execute("INSERT INTO champions (task,modelId,experimentId,promotedAt,active,reason,doc) "
                  "VALUES (?,?,?,?,1,?,?)",
                  (task, target["modelId"], target["experimentId"], now(),
                   f"Rollback: restored previous champion {target['modelId']}", target["doc"]))
    return get_champion(task)


# ------------------------------------------------------------------ config
@app.get("/config")
def get_config():
    with conn() as c:
        return {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT * FROM config")}


@app.put("/config")
def put_config(values: dict = Body(...)):
    with conn() as c:
        for k, v in values.items():
            c.execute("INSERT OR REPLACE INTO config VALUES (?,?)", (k, json.dumps(v)))
    return get_config()
