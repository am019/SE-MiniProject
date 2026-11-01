"""Integration / system tests on the real multi-process platform (TC-06 ... TC-19, TC-26 ... TC-28, TC-31)."""
import hashlib
import json
import re
import time
from pathlib import Path

import httpx
import jwt
import pytest

from tests.conftest import FAST, ROOT, vol_body

pytestmark = pytest.mark.integration


def svc(platform, name, method, path, **kw):
    return httpx.request(method, platform.url(name) + path, headers=platform.service_headers, timeout=300, **kw)


def train_register(platform, name, model, fs_id, config=None, seed=42):
    r = svc(platform, name, "POST", "/train", json={"featureSetId": fs_id, "model": model, "config": config or {},
                                                    "seed": seed, "timeoutSec": 300})
    if r.status_code != 200:
        return r
    j = r.json()
    svc(platform, "registry", "POST", "/models", json={"modelId": j["modelId"], "name": model, "family": name,
                                                       "status": "COMPLETED", "artifactHash": j["artifactHash"]})
    return r


@pytest.fixture(scope="module")
def fs_vol(platform, vol_run):
    r = svc(platform, "features", "POST", "/featuresets", json={"datasetId": vol_run["ds"]["datasetId"],
                                                              "task": "VOLATILITY", "horizon": 5})
    assert r.status_code == 200
    return r.json()


def predict(platform, svc_name, model_id, fs_id, split="test"):
    return svc(platform, svc_name, "POST", "/predict", json={"modelId": model_id, "featureSetId": fs_id, "split": split})


# ------------------------------------------------------------------- model contracts
def test_tc06_classical_service_contract(platform, fs_vol):
    d = svc(platform, "classical", "GET", "/describe").json()
    assert d["family"] == "classical" and {"garch", "random_forest"} <= {m["name"] for m in d["models"]}
    assert httpx.get(platform.url("classical") + "/health").json()["status"] == "UP"
    for model in ("garch", "random_forest"):
        t = train_register(platform, "classical", model, fs_vol["featureSetId"], seed=7)
        assert t.status_code == 200, t.text
        j = t.json()
        assert j["status"] == "COMPLETED" and j["seed"] == 7 and j["trainTimeSec"] > 0 and j["config"] is not None
        p = predict(platform, "classical", j["modelId"], fs_vol["featureSetId"]).json()
        assert len(p["values"]) == fs_vol["splitRanges"]["test"]["rows"] and p["inferenceTimeMs"] >= 0
        assert all(v > 0 for v in p["values"])
    bad = svc(platform, "classical", "POST", "/train", json={"featureSetId": fs_vol["featureSetId"], "model": "random_forest",
                                                             "config": {"n_estimators": 5}})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "CONFIG_INVALID"
    assert svc(platform, "classical", "POST", "/predict", json={"modelId": "nope-1", "featureSetId": fs_vol["featureSetId"]}).status_code == 404


def test_tc07_deep_learning_service_contract(platform, fs_vol):
    d = svc(platform, "dl", "GET", "/describe").json()
    assert d["family"] == "dl" and d["models"][0]["name"] == "lstm"
    t = train_register(platform, "dl", "lstm", fs_vol["featureSetId"], {"epochs": 8, "patience": 3}, seed=11)
    assert t.status_code == 200, t.text
    j = t.json()
    assert j["seed"] == 11 and j["info"]["epochsRun"] <= 8 and "patience 3" in j["info"]["earlyStopping"]
    p = predict(platform, "dl", j["modelId"], fs_vol["featureSetId"]).json()
    assert len(p["values"]) == fs_vol["splitRanges"]["test"]["rows"]


def test_tc08_quantum_service_and_qubit_limit(platform, fs_vol):
    d = svc(platform, "qml", "GET", "/describe").json()
    assert d["maxQubits"] == 8
    ok = train_register(platform, "qml", "qml_vqr", fs_vol["featureSetId"], {"n_qubits": 4, "epochs": 10})
    assert ok.status_code == 200, ok.text
    info = ok.json()["info"]
    assert info["nQubits"] == 4 and info["simulator"] == "default.qubit" and "simulator" in info["device"]
    r = svc(platform, "qml", "POST", "/train", json={"featureSetId": fs_vol["featureSetId"], "model": "qml_vqr",
                                                     "config": {"n_qubits": 9}})
    e = r.json()["error"]
    assert r.status_code == 422 and e["code"] == "QUBIT_LIMIT_EXCEEDED" and "8" in e["message"]


# ------------------------------------------------------------------- experiments
def test_tc09_experiment_configuration_and_run(platform, vol_run):
    e = platform.client().get(f"/api/experiments/{vol_run['id']}").json()
    assert (e["task"], e["horizon"], e["seed"], e["primaryMetric"]) == ("VOLATILITY", 5, 42, "RMSE")
    assert e["baseline"] == "garch" and e["baselineAutoAdded"] is True and "garch" in e["models"]
    seen = vol_run["exp"]["_seen"]
    assert any("TRAINING" in s.values() for s in seen), "per-model status must progress"
    assert vol_run["exp"]["status"] == "COMPLETED"
    c = platform.client()
    for body, code in [({"datasetId": vol_run["ds"]["datasetId"], "task": "VOLATILITY", "models": ["nope"]}, 422),
                       ({"datasetId": vol_run["ds"]["datasetId"], "task": "VOLATILITY", "models": ["random_forest"], "horizon": 0}, 422),
                       ({"datasetId": vol_run["ds"]["datasetId"], "task": "VOLATILITY", "models": ["random_forest"], "primaryMetric": "BOGUS"}, 422),
                       ({"datasetId": "ds-000000000000", "task": "VOLATILITY", "models": ["random_forest"]}, 404),
                       ({"datasetId": vol_run["ds"]["datasetId"], "task": "OPTION_PRICE", "models": ["random_forest"]}, 422)]:
        r = c.post("/api/experiments", json=body)
        assert r.status_code == code, (body, r.text)
    assert c.get("/api/experiments/exp-9999").status_code == 404


def test_tc10_identical_test_timestamps_across_models(platform, vol_run):
    lb = platform.client().get(f"/api/experiments/{vol_run['id']}/leaderboard").json()
    done = [r for r in lb["leaderboard"] if r["status"] == "COMPLETED"]
    assert len(done) == 5
    assert len({r["testTimestampHash"] for r in done}) == 1 and done[0]["testTimestampHash"] == lb["testTimestampHash"]
    assert len({r["featureSetVersion"] for r in done}) == 1
    sel = platform.client().get(f"/api/experiments/{vol_run['id']}/selection").json()
    assert sel["integrity"]["identicalTestTimestampHash"] and sel["integrity"]["identicalFeatureSetVersion"]
    for r in done:
        assert set(r["val"]) >= {"RMSE", "MAE", "QLIKE", "MAPE", "DA"} and r["test"]["RMSE"] > 0


def test_tc11_leaderboard_display_data(platform, vol_run):
    lb = platform.client().get(f"/api/experiments/{vol_run['id']}/leaderboard").json()
    assert {"testPeriod", "valPeriod", "primaryMetric", "policy", "metricDefinitions"} <= set(lb)
    fams = {r["family"] for r in lb["leaderboard"]}
    assert fams == {"classical", "dl", "qml"}
    ranks = [r["rank"] for r in lb["leaderboard"]]
    assert ranks == sorted(ranks)
    vals = [r["val"]["RMSE"] for r in lb["leaderboard"]]
    assert vals == sorted(vals)                          # ranked by validation primary metric
    for r in lb["leaderboard"]:
        assert r["trainTimeSec"] > 0 and r["inferenceTimeMs"] >= 0 and r["status"] == "COMPLETED"
    assert sum(r["isChampion"] for r in lb["leaderboard"]) == 1


def test_tc12_selection_in_platform_uses_validation_only(platform, vol_run):
    s = platform.client().get(f"/api/experiments/{vol_run['id']}/selection").json()
    lb = platform.client().get(f"/api/experiments/{vol_run['id']}/leaderboard").json()["leaderboard"]
    base = next(r for r in lb if r["isBaseline"])
    elig = [r for r in lb if not r["isBaseline"] and r["val"]["RMSE"] < base["val"]["RMSE"]]
    if elig:
        best = min(r["val"]["RMSE"] for r in elig)
        near = [r for r in elig if r["val"]["RMSE"] <= best * 1.01]
        want = min(near, key=lambda r: r["inferenceTimeMs"])["model"]
        assert s["champion"]["model"] == want
    else:
        assert s["champion"]["model"] == "garch" and "NO_MODEL_BEAT_BASELINE" in s["flags"]
    assert s["justification"] and s["policy"]["selectionData"] == "validation only"


def test_tc14_derivatives_task_end_to_end(platform, opt_run):
    e = opt_run["exp"]
    assert e["task"] == "OPTION_PRICE" and e["baseline"] == "black_scholes" and e["primaryMetric"] == "MAE"
    lb = platform.client().get(f"/api/experiments/{opt_run['id']}/leaderboard").json()
    done = [r for r in lb["leaderboard"] if r["status"] == "COMPLETED"]
    assert {r["family"] for r in done} == {"classical", "dl", "qml"} and len(done) == 4
    assert len({r["testTimestampHash"] for r in done}) == 1
    assert "QLIKE" not in done[0]["val"] and done[0]["val"]["DA"] is None
    assert any(r["isChampion"] for r in done)


def test_tc15_prediction_with_champion(platform, vol_run, opt_run):
    c = platform.client()
    ch = c.get("/api/models/champion", params={"task": "VOLATILITY"}).json()
    t0 = time.perf_counter()
    r = c.post("/api/predictions", json={"task": "VOLATILITY", "horizon": 5})
    dt = time.perf_counter() - t0
    p = r.json()
    assert r.status_code == 200 and dt < 2.0, (r.text, dt)
    assert p["modelId"] == ch["modelId"] and p["family"] and p["version"] and p["datasetId"]
    assert p["horizon"] == 5 and p["validationMetric"]["name"] == "RMSE" and p["validationMetric"]["value"] > 0
    assert 0.0 < p["values"][0] < 3 and "not investment advice" in p["disclaimer"] and p["isSample"] is True
    bad = c.post("/api/predictions", json={"task": "VOLATILITY", "horizon": 9})
    assert bad.status_code == 422 and "horizon" in bad.json()["error"]["message"]
    o = c.post("/api/predictions", json={"task": "OPTION_PRICE", "strike": 5000, "daysToExpiry": 30, "optionType": "call"})
    assert o.status_code == 200 and o.json()["values"][0] > 0
    assert c.post("/api/predictions", json={"task": "OPTION_PRICE"}).status_code == 422


def test_tc18_report_export_and_provenance(platform, vol_run):
    c, eid = platform.client(), vol_run["id"]
    j = c.get(f"/api/reports/{eid}", params={"format": "json"})
    assert j.status_code == 200 and "attachment" in j.headers["content-disposition"]
    rep = j.json()
    assert rep["selection"]["justification"] and rep["limitations"] and rep["configuration"]["baseline"] == "garch"
    models = {m["modelId"]: m for m in c.get("/api/models", params={"experimentId": eid}).json()["models"]}
    ds = c.get(f"/api/datasets/{vol_run['ds']['datasetId']}").json()
    assert rep["dataset"]["datasetHash"] == ds["datasetHash"]
    for row in rep["leaderboard"]:
        assert row["artifactHash"] == models[row["modelId"]]["artifactHash"] and len(row["artifactHash"]) == 64
    csvr = c.get(f"/api/reports/{eid}", params={"format": "csv"})
    assert csvr.headers["content-type"].startswith("text/csv") and "artifactSha256" in csvr.text.splitlines()[0]
    assert rep["leaderboard"][0]["artifactHash"] in csvr.text
    pdf = c.get(f"/api/reports/{eid}", params={"format": "pdf"})
    assert pdf.content[:5] == b"%PDF-" and len(pdf.content) > 2000
    assert c.get(f"/api/reports/{eid}", params={"format": "xml"}).status_code == 422
    assert c.get("/api/reports/exp-9999", params={"format": "json"}).status_code == 404


def test_tc19_overview_health_and_correlation_id(platform, vol_run):
    c = platform.client()
    t0 = time.perf_counter()
    h = c.get("/api/admin/health").json()
    names = {s["service"] for s in h["services"]}
    assert {"gateway", "registry", "ingestion", "features", "evaluation", "report", "orchestrator", "classical", "dl", "qml"} <= names
    assert h["overall"] == "UP" and all(s["lastChecked"] and s["latencyMs"] < 1000 for s in h["services"])
    assert time.perf_counter() - t0 < 3
    for n in ("registry", "ingestion", "classical"):
        t = time.perf_counter()
        assert httpx.get(platform.url(n) + "/health", timeout=1).status_code == 200
        assert time.perf_counter() - t < 1.0
    # one request id must appear in the logs of every service the request touches
    rid = "req-tc19-" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    r = c.post("/api/experiments", json=vol_body(vol_run["ds"], ["random_forest"]), headers={"X-Request-ID": rid})
    assert r.status_code == 202 and r.headers["x-request-id"] == rid
    eid = r.json()["experimentId"]
    for _ in range(120):
        if c.get(f"/api/experiments/{eid}").json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(1)
    for name in ("gateway", "orchestrator", "ingestion", "features", "registry", "classical", "evaluation"):
        assert rid in platform.logs(name), f"{rid} missing from {name} log"
    assert c.get("/").status_code == 200 and "IFAP" in c.get("/").text          # dashboard served by the gateway


# ------------------------------------------------------------------- security (shared platform)
def test_tc26_authentication_and_rbac(platform):
    base = platform.base
    assert httpx.get(base + "/api/experiments").status_code == 401
    assert httpx.get(base + "/api/experiments", headers={"Authorization": "Bearer garbage"}).status_code == 401
    secret = platform.secrets["IFAP_JWT_SECRET"]
    expired = jwt.encode({"sub": "admin", "role": "Administrator", "iat": 1, "exp": int(time.time()) - 60}, secret, algorithm="HS256")
    r = httpx.get(base + "/api/experiments", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHENTICATED"
    forged = jwt.encode({"sub": "admin", "role": "Administrator", "exp": int(time.time()) + 600}, "wrong-secret", algorithm="HS256")
    assert httpx.get(base + "/api/admin/health", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    assert httpx.post(base + "/api/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    an, ad = platform.client("analyst"), platform.client("admin")
    for method, path in [("GET", "/api/admin/health"), ("GET", "/api/admin/config"),
                         ("POST", "/api/models/champion/rollback"), ("DELETE", "/api/models/x-1")]:
        r = an.request(method, path, json={"task": "VOLATILITY"} if method == "POST" else None)
        assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN", (path, r.text)
    assert ad.get("/api/admin/health").status_code == 200
    assert an.get("/api/experiments").status_code == 200       # Analyst keeps normal access
    assert httpx.get(base + "/health").status_code == 200       # public probe


def test_tc27_network_isolation_and_service_tokens(platform):
    for name in ("registry", "orchestrator", "ingestion", "classical", "evaluation", "features", "report"):
        r = httpx.get(platform.url(name) + "/experiments" if name in ("registry", "orchestrator") else platform.url(name) + "/describe" if name in ("classical",) else platform.url(name) + "/datasets" if name == "ingestion" else platform.url(name) + "/featuresets/x")
        assert r.status_code == 401, (name, r.status_code)
        r = httpx.get(platform.url(name) + "/health")
        assert r.status_code == 200                                # only the health probe is open
    r = httpx.get(platform.url("registry") + "/experiments", headers={"X-Service-Token": "wrong"})
    assert r.status_code == 401
    assert httpx.get(platform.url("registry") + "/experiments", headers=platform.service_headers).status_code == 200
    # the gateway never exposes internal paths
    assert httpx.get(platform.base + "/registry/experiments").status_code in (401, 404)


def test_tc28_gateway_upload_rules(platform):
    c = platform.client()
    f = lambda name, data, ct="text/csv": {"file": (name, data, ct)}  # noqa: E731
    r = c.post("/api/datasets", files=f("evil.exe", b"MZ\x90\x00"))
    assert r.status_code == 415 and r.json()["error"]["code"] == "FILE_TYPE_UNSUPPORTED"
    assert c.post("/api/datasets", files=f("sheet.xlsx", b"PK\x03\x04")).status_code == 415
    big = b"date,close\n" + b"2020-01-01,1\n" * (26 * 1024 * 1024 // 13)
    r = c.post("/api/datasets", files=f("big.csv", big))
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
    body = b"date,close,note\n" + b"".join(f"2021-{1 + i // 28:02d}-{1 + i % 28:02d},{100 + i},<b>=1+1</b>\n".encode() for i in range(130))
    r = c.post("/api/datasets", files=f("cells.csv", body))
    assert r.status_code == 201 and "<b>" not in r.text and "=1+1" not in r.text
    js = (ROOT / "web" / "app.js").read_text()
    assert "innerHTML" not in js and "document.write" not in js and "eval(" not in js   # cells rendered via textContent only
    csp = httpx.get(platform.base + "/").headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp


def test_tc31_safe_error_messages(platform):
    c = platform.client()
    leaks = re.compile(r"Traceback|File \"|\.py\b|/home/|/tmp/|site-packages|IFAP_|secret", re.I)
    bad = [c.post("/api/datasets", files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")}),
           c.post("/api/experiments", json={"nonsense": True}),
           c.get("/api/datasets/ds-doesnotexist"), c.get("/api/nope"),
           c.post("/api/predictions", json={"task": "BOGUS"}),
           c.post("/api/experiments", content=b"{bad json", headers={"Content-Type": "application/json"})]
    for r in bad:
        assert r.status_code >= 400
        e = r.json()["error"]
        assert set(e) <= {"code", "message", "requestId", "fields"} and e["requestId"] and not leaks.search(r.text), r.text
    assert httpx.get(platform.base + "/api/experiments").json()["error"]["code"] == "UNAUTHENTICATED"
