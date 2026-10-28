"""Tests that need their own platform instance (stopped services, faults, tampering, rate limit, stub)."""
import json
import re
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import FAST, ROOT, Platform, vol_body

pytestmark = pytest.mark.integration
CLASSICAL_ONLY = "registry,ingestion,features,evaluation,report,orchestrator,classical"


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory):
    p = Platform(tmp_path_factory.mktemp("ifap_life"), only=CLASSICAL_ONLY)
    yield p
    p.stop()


def champion(p, task="VOLATILITY"):
    return p.client().get("/api/models/champion", params={"task": task}).json()["modelId"]


# -------------------------------------------------------------------- TC-17, TC-30
def test_tc17_registry_retrain_failure_and_rollback(lifecycle):
    p, c = lifecycle, lifecycle.client()
    ds = p.sample()
    assert c.get("/api/models/champion", params={"task": "VOLATILITY"}).json()["error"]["code"] == "NO_CHAMPION"
    a = p.run(vol_body(ds, ["random_forest"]))
    champ_a = champion(p)
    # a new run fails entirely -> champion must not change (FR-19)
    b = p.run(vol_body(ds, ["garch", "random_forest"], modelConfig={"garch": {"fault": "crash"}, "random_forest": {"fault": "crash"}}),
              expect="FAILED")
    assert b["status"] == "FAILED" and champion(p) == champ_a
    failed = c.get("/api/models", params={"experimentId": b["experimentId"]}).json()["models"]
    assert len(failed) == 2 and all(m["status"] == "FAILED" and m["failureReason"] for m in failed)   # runs recorded (FR-16)
    # promotion only after completion: while a good run is in flight the old champion stays
    r = c.post("/api/experiments", json=vol_body(ds, ["random_forest", "gradient_boosting"], horizon=10, seed=7))
    assert r.status_code == 202 and champion(p) == champ_a
    r2 = c.post("/api/experiments", json=vol_body(ds, ["random_forest"]))
    assert r2.status_code == 409 and r2.json()["error"]["code"] == "EXPERIMENT_IN_PROGRESS"
    eid = r.json()["experimentId"]
    for _ in range(300):
        if c.get(f"/api/experiments/{eid}").json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(1)
    champ_c = champion(p)
    assert champ_c != champ_a
    hist = c.get("/api/models/champion/history", params={"task": "VOLATILITY"}).json()["history"]
    assert [h["modelId"] for h in hist][:2] == [champ_c, champ_a] and sum(h["active"] for h in hist) == 1
    # roll back (admin) restores the previous champion; its record and history stay intact
    rb = c.post("/api/models/champion/rollback", json={"task": "VOLATILITY"})
    assert rb.status_code == 200 and champion(p) == champ_a
    assert len(c.get("/api/models/champion/history", params={"task": "VOLATILITY"}).json()["history"]) == 3
    assert c.post("/api/models/champion/rollback", json={"task": "OPTION_PRICE"}).json()["error"]["code"] == "NO_PREVIOUS_CHAMPION"
    assert c.delete(f"/api/models/{champ_a}").status_code == 409          # active champion cannot be deleted
    assert c.get(f"/api/models/{champ_c}").json()["modelId"] == champ_c


def test_tc30_artifact_tamper_is_refused(lifecycle):
    p, c = lifecycle, lifecycle.client()
    champ = champion(p)
    before = c.post("/api/predictions", json={"task": "VOLATILITY"})
    assert before.status_code == 200
    files = [f for f in (p.data / "models" / champ).iterdir() if f.is_file()]
    f = files[0]
    orig = f.read_bytes()
    try:
        f.write_bytes(bytes([orig[0] ^ 0x01]) + orig[1:])           # flip one bit
        r = c.post("/api/predictions", json={"task": "VOLATILITY"})
        assert r.status_code == 500 and r.json()["error"]["code"] == "ARTIFACT_INTEGRITY_FAILED"
        assert champion(p) == champ                                  # champion pointer unchanged
        assert "artifact hash mismatch" in p.logs("classical") and champ in p.logs("classical")   # operator alert (log)
    finally:
        f.write_bytes(orig)
    again = c.post("/api/predictions", json={"task": "VOLATILITY"})
    assert again.status_code == 200 and again.json()["values"] == before.json()["values"]


def test_tc31_model_exception_is_safe(lifecycle):
    p, c = lifecycle, lifecycle.client()
    ds = p.sample()
    e = p.run(vol_body(ds, ["random_forest"], modelConfig={"random_forest": {"fault": "crash"}}))   # garch baseline succeeds
    leaks = re.compile(r"Traceback|File \"|\.py\b|/home/|/tmp/|site-packages|injected fault", re.I)
    assert e["models"]["random_forest"]["status"] == "FAILED" and not leaks.search(json.dumps(e))
    lb = c.get(f"/api/experiments/{e['experimentId']}/leaderboard").text
    assert not leaks.search(lb)
    assert "isolated job failed" in p.logs("classical")             # detail is available only in the restricted log


def test_tc22_reproducibility_classical_identical_5_of_5(lifecycle):
    p = lifecycle
    ds = p.sample()
    runs = []
    for _ in range(5):
        e = p.run(vol_body(ds, ["random_forest", "gradient_boosting"], seed=42))
        lb = p.client().get(f"/api/experiments/{e['experimentId']}/leaderboard").json()
        runs.append((e["splitHash"], e["testTimestampHash"], e["featureSetId"],
                     {r["model"]: (r["val"], r["test"]) for r in lb["leaderboard"]}))
    assert all(r == runs[0] for r in runs[1:]), "split timestamps and classical metrics must be identical 5/5"


def test_tc22_dl_qml_variation_is_measured(platform):
    """DL / QML: repeat with the same seed and with different seeds; the spread is recorded, not asserted to be zero."""
    ds = platform.sample()
    rows = {"same_seed": {"lstm": [], "qml_vqr": []}, "other_seeds": {"lstm": [], "qml_vqr": []}}
    for kind, seeds in (("same_seed", [42, 42, 42]), ("other_seeds", [1, 2, 3])):
        for s in seeds:
            e = platform.run(vol_body(ds, ["lstm", "qml_vqr"], seed=s))
            lb = platform.client().get(f"/api/experiments/{e['experimentId']}/leaderboard").json()["leaderboard"]
            for r in lb:
                if r["model"] in rows[kind]:
                    rows[kind][r["model"]].append(r["val"]["RMSE"])
    print("REPRODUCIBILITY", json.dumps(rows))
    for kind in rows:
        for vals in rows[kind].values():
            assert len(vals) == 3 and all(v > 0 for v in vals)
    for m, vals in rows["same_seed"].items():
        assert max(vals) - min(vals) < 0.02, f"same-seed variation of {m} unexpectedly large"


# -------------------------------------------------------------------- TC-16
def test_tc16_fault_isolation_with_stopped_qml_3_of_3(tmp_path_factory):
    p = Platform(tmp_path_factory.mktemp("ifap_noqml"), only=CLASSICAL_ONLY + ",dl")
    try:
        ds = p.sample()
        for i in range(3):
            e = p.run(vol_body(ds, ["random_forest", "lstm", "qml_vqr"]))
            assert e["models"]["qml_vqr"]["status"] == "FAILED" and "SERVICE_UNAVAILABLE" in e["models"]["qml_vqr"]["reason"]
            assert all(e["models"][m]["status"] == "COMPLETED" for m in ("garch", "random_forest", "lstm"))
            lb = p.client().get(f"/api/experiments/{e['experimentId']}/leaderboard").json()
            q = next(r for r in lb["leaderboard"] if r["model"] == "qml_vqr")
            assert q["rank"] is None and not q["eligible"] and "excluded" in q["note"]
            sel = p.client().get(f"/api/experiments/{e['experimentId']}/selection").json()
            assert sel["champion"]["model"] != "qml_vqr" and any(x["model"] == "qml_vqr" for x in sel["excluded"])
            assert len(lb["leaderboard"]) == 4
    finally:
        p.stop()


# -------------------------------------------------------------------- TC-21 (+ SR-07 job limits)
@pytest.mark.slow
def test_tc21_time_bounds_and_job_limits(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_sample_data.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    p = Platform(tmp_path_factory.mktemp("ifap_limits"), only=CLASSICAL_ONLY + ",dl", env={"IFAP_JOB_MEM_MB": "6144"})
    try:
        c = p.client()
        big = mk.price_series(n=5000, seed=3).to_csv(index=False).encode()
        ds = c.post("/api/datasets", files={"file": ("big5000.csv", big, "text/csv")}).json()
        assert ds["status"] == "VALID" and ds["rows"] == 5000
        e = p.run({"datasetId": ds["datasetId"], "task": "VOLATILITY", "horizon": 5, "seed": 42,
                   "models": ["random_forest", "gradient_boosting", "lstm"]}, timeout=1500)
        lb = {r["model"]: r for r in c.get(f"/api/experiments/{e['experimentId']}/leaderboard").json()["leaderboard"]}
        print("TC-21 5,000 rows train seconds:", {k: v["trainTimeSec"] for k, v in lb.items()})
        for m in ("garch", "random_forest", "gradient_boosting"):
            assert lb[m]["trainTimeSec"] <= 300
        assert lb["lstm"]["trainTimeSec"] <= 900
        # an endless job and a memory hog are stopped; the rest of the platform keeps working
        small = p.sample()
        e = p.run(vol_body(small, ["random_forest", "gradient_boosting"],
                           modelConfig={"gradient_boosting": {"fault": "hang"}, "random_forest": {"fault": "oom"}}, timeoutSec=25))
        assert e["models"]["gradient_boosting"]["status"] == "TIMED_OUT" and e["models"]["gradient_boosting"]["reason"] == "TIMEOUT"
        assert e["models"]["random_forest"]["status"] == "FAILED"
        assert e["models"]["garch"]["status"] == "COMPLETED"
        h = c.get("/api/admin/health").json()
        assert h["overall"] in ("UP", "DEGRADED") and all(s["status"] == "UP" for s in h["services"] if s["service"] in ("classical", "gateway", "registry"))
    finally:
        p.stop()


# -------------------------------------------------------------------- TC-23
def test_tc23_extensibility_stub_by_configuration_only(tmp_path_factory):
    for part in ("gateway", "orchestrator", "evaluation"):
        src = "".join(f.read_text() for f in (ROOT / "services" / part).glob("*.py"))
        assert "stub" not in src.lower(), f"{part} must not know about the stub model"
    assert "stub" not in "".join((ROOT / "web" / n).read_text() for n in ("app.js", "index.html", "style.css")).lower()
    p = Platform(tmp_path_factory.mktemp("ifap_stub"), only=CLASSICAL_ONLY + ",stub",
                 models_config=str(ROOT / "config" / "models.stub.json"))
    try:
        c = p.client()
        av = c.get("/api/models/available").json()
        assert any(m["name"] == "stub_persistence" and m["serviceStatus"] == "UP" for m in av["models"])
        e = p.run(vol_body(p.sample(), ["random_forest", "stub_persistence"]))
        lb = c.get(f"/api/experiments/{e['experimentId']}/leaderboard").json()["leaderboard"]
        s = next(r for r in lb if r["model"] == "stub_persistence")
        assert s["status"] == "COMPLETED" and s["family"] == "stub" and s["val"]["RMSE"] > 0
        ds = p.sample("sample_option_chain.csv")
        e2 = p.run({"datasetId": ds["datasetId"], "task": "OPTION_PRICE", "models": ["stub_persistence"], "seed": 1})
        assert e2["models"]["stub_persistence"]["status"] == "COMPLETED"
    finally:
        p.stop()


# -------------------------------------------------------------------- TC-32
def test_tc32_rate_limiting_per_user(tmp_path_factory):
    p = Platform(tmp_path_factory.mktemp("ifap_rate"), only="registry", rate_limit=60)
    try:
        an, ad = p.client("analyst"), p.client("admin")
        codes = [an.get("/api/me").status_code for _ in range(70)]
        assert codes[:60] == [200] * 60 and set(codes[60:]) == {429}
        r = an.get("/api/me")
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1 and r.json()["error"]["code"] == "RATE_LIMITED"
        assert ad.get("/api/me").status_code == 200                  # other users unaffected
        assert httpx.get(p.base + "/health").status_code == 200       # probes are not rate limited
    finally:
        p.stop()
