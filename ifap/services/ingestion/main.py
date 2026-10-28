"""Ingestion & Validation service (FR-01, FR-02, FR-03, SR-03, NFR-01)."""
from __future__ import annotations

import json
import re
import time

import numpy as np
from fastapi import Request

from ifap_common import datasets as D
from ifap_common.appfactory import create_app
from ifap_common.config import env_int
from ifap_common.errors import AppError
from ifap_common.featureset import sha256_bytes

app = create_app("ingestion")
MAX_BYTES = env_int("IFAP_MAX_UPLOAD_MB", 25) * 1024 * 1024


def _safe_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", (name or "dataset.csv").split("/")[-1].split("\\")[-1])
    return base[:80] or "dataset.csv"


@app.post("/datasets")
async def upload(request: Request, filename: str = "dataset.csv"):
    """Raw body upload (the gateway forwards the validated multipart payload)."""
    name = _safe_name(filename)
    if not name.lower().endswith(".csv"):
        raise AppError("FILE_TYPE_UNSUPPORTED", "Only .csv files are accepted.")  # before parsing (SR-03)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BYTES:
            raise AppError("FILE_TOO_LARGE", f"File exceeds the {MAX_BYTES // (1024 * 1024)} MB limit.")
    raw = bytes(raw)
    t0 = time.time()
    res = D.validate(raw)  # raises SCHEMA_INVALID when required columns are missing
    h = sha256_bytes(raw)
    ds_id = "ds-" + h[:12]
    status = "BLOCKED" if res["blocking"] else "VALID"
    meta = {"datasetId": ds_id, "name": name, "datasetHash": h, "type": res["type"], "status": status,
            "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "isSample": name.lower().startswith("sample_"), **res["stats"],
            "issues": res["issues"], "validationSeconds": round(time.time() - t0, 3)}
    (D.ds_dir() / f"{ds_id}.csv").write_bytes(raw)
    (D.ds_dir() / f"{ds_id}.json").write_text(json.dumps(meta))
    return _public(meta) | {"validationReport": {"status": status, "issues": res["issues"],
                                                    "blockingIssues": sum(i["severity"] == "BLOCKING"
                                                                          for i in res["issues"])}}


def _public(m: dict) -> dict:
    keys = ["datasetId", "name", "datasetHash", "type", "status", "uploadedAt", "rows", "columns",
            "dateMin", "dateMax", "ignoredColumns", "validationSeconds", "isSample"]
    d = {k: m.get(k) for k in keys}
    d["blockingIssues"] = sum(i["severity"] == "BLOCKING" for i in m.get("issues", []))
    d["warnings"] = sum(i["severity"] == "WARNING" for i in m.get("issues", []))
    return d


@app.get("/datasets")
def list_datasets():
    out = []
    for p in sorted(D.ds_dir().glob("ds-*.json")):
        try:
            out.append(_public(json.loads(p.read_text())))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda m: m["uploadedAt"] or "", reverse=True)
    return {"datasets": out}


@app.get("/datasets/{ds_id}")
def get_dataset(ds_id: str):
    m = D.load_meta(ds_id)
    return _public(m) | {"validationReport": {"status": m["status"], "issues": m["issues"]}}


@app.get("/datasets/{ds_id}/summary")
def summary(ds_id: str):
    meta = D.load_meta(ds_id)
    res = D.validate((D.ds_dir() / f"{ds_id}.csv").read_bytes())
    c = res["clean"]
    out = _public(meta) | {"validationStatus": meta["status"]}
    cols = {}
    for col in c.columns:
        if str(c[col].dtype).startswith(("float", "int")):
            s = c[col].dropna()
            if len(s):
                cols[col] = {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
                             "std": float(s.std()) if len(s) > 1 else 0.0, "count": int(len(s))}
    out["columns_summary"] = cols
    if meta["type"] == "OPTION_CHAIN":
        out["distinctDates"] = int(c["date"].nunique())
        out["optionTypes"] = {k: int(v) for k, v in c["option_type"].value_counts().items()}
    else:
        r = np.log(c["close"].dropna()).diff().dropna()
        out["returnsSummary"] = {"count": int(len(r)),
                                 "annualisedVolatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else None}
    return out
