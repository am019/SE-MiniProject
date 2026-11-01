"""Test fixtures: an in-process app set and a real multi-process platform (scripts/run_local.py)."""
from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]
SAMPLE = ROOT / "data" / "sample"
FAST = {"lstm": {"epochs": 12, "patience": 3}, "qml_vqr": {"epochs": 30}}  # keep CI runs short


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: starts the real multi-process platform")
    config.addinivalue_line("markers", "slow: long-running performance tests")


# --------------------------------------------------------------------- in-process
@pytest.fixture()
def inproc(tmp_path, monkeypatch):
    """Ingestion + features apps in this process (no network), isolated data dir."""
    monkeypatch.setenv("IFAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IFAP_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("IFAP_MAX_UPLOAD_MB", "25")
    import ifap_common.config as cfg
    cfg._cache.clear()
    from fastapi.testclient import TestClient
    from services.features.main import app as fa
    from services.ingestion.main import app as ia
    h = {"X-Service-Token": "test-token"}
    ic, fc = TestClient(ia, headers=h, raise_server_exceptions=False), TestClient(fa, headers=h, raise_server_exceptions=False)

    class NS:
        ingest, features, data = ic, fc, tmp_path

        @staticmethod
        def upload(raw: bytes, name="fixture.csv"):
            return ic.post("/datasets", params={"filename": name}, content=raw)

        @staticmethod
        def build(ds_id, task="VOLATILITY", **kw):
            return fc.post("/featuresets", json={"datasetId": ds_id, "task": task, **kw})
    return NS


# --------------------------------------------------------------------- platform
def _free_base(n=14):
    for _ in range(50):
        base = random.randint(20000, 40000)
        socks = []
        try:
            for p in range(base, base + n + 1):
                s = socket.socket()
                s.bind(("127.0.0.1", p))
                socks.append(s)
            return base
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    raise RuntimeError("no free port block")


class Platform:
    def __init__(self, tmp: Path, only="", models_config=None, env=None, rate_limit=100000):
        self.data = tmp
        self.base_port = _free_base()
        self.gw = self.base_port + 13
        cfgp = models_config or str(ROOT / "config" / "models.json")
        names = ["registry", "ingestion", "features", "evaluation", "report", "orchestrator"] + \
            list(json.loads(Path(cfgp).read_text())["services"])
        self.ports = {n: self.base_port + i for i, n in enumerate(names)}
        e = dict(os.environ, IFAP_ALLOW_FAULT_INJECTION="1", IFAP_RATE_LIMIT=str(rate_limit), **(env or {}))
        cmd = [sys.executable, str(ROOT / "scripts" / "run_local.py"), "--data-dir", str(tmp), "--port", str(self.gw),
               "--base-port", str(self.base_port), "--models-config", cfgp]
        if only:
            cmd += ["--only", only]
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=e, stdout=open(tmp / "run.out", "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 180:
            if "IFAP is running" in (tmp / "run.out").read_text():
                break
            if self.proc.poll() is not None:
                raise RuntimeError("platform exited: " + (tmp / "run.out").read_text()[-800:])
            time.sleep(0.5)
        else:
            raise RuntimeError("platform did not start: " + (tmp / "run.out").read_text()[-800:])
        self.secrets = json.loads((tmp / ".secrets.json").read_text())
        self.base = f"http://127.0.0.1:{self.gw}"
        self._clients = {}

    def url(self, name):
        return f"http://127.0.0.1:{self.ports[name]}"

    @property
    def service_headers(self):
        return {"X-Service-Token": self.secrets["IFAP_SERVICE_TOKEN"]}

    def client(self, user="admin") -> httpx.Client:
        if user not in self._clients:
            pw = self.secrets["IFAP_ADMIN_PASSWORD" if user == "admin" else "IFAP_ANALYST_PASSWORD"]
            c = httpx.Client(base_url=self.base, timeout=120)
            tok = c.post("/api/auth/login", json={"username": user, "password": pw}).json()["token"]
            c.headers["Authorization"] = f"Bearer {tok}"
            self._clients[user] = c
        return self._clients[user]

    def sample(self, name="sample_index_daily.csv", user="admin"):
        r = self.client(user).post(f"/api/samples/{name}/load")
        assert r.status_code == 201, r.text
        return r.json()

    def run(self, body, user="admin", timeout=600, expect="COMPLETED"):
        c = self.client(user)
        r = c.post("/api/experiments", json=body)
        assert r.status_code == 202, r.text
        eid = r.json()["experimentId"]
        t0, seen = time.time(), []
        while time.time() - t0 < timeout:
            e = c.get(f"/api/experiments/{eid}").json()
            seen.append({k: v["status"] for k, v in e["models"].items()})
            if e["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(1)
        assert e["status"] == expect, e
        e["_seen"] = seen
        return e

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def logs(self, name):
        p = self.data / "logs" / f"{name}.log"
        return p.read_text() if p.exists() else ""


@pytest.fixture(scope="session")
def platform(tmp_path_factory):
    p = Platform(tmp_path_factory.mktemp("ifap_full"))
    yield p
    p.stop()


def vol_body(ds, models, **kw):
    return {"datasetId": ds["datasetId"], "task": "VOLATILITY", "horizon": 5, "models": models, "seed": 42,
            "modelConfig": {m: FAST[m] for m in models if m in FAST}, **kw}


@pytest.fixture(scope="session")
def vol_run(platform):
    """One full 3-family experiment on the volatility sample (shared by many TCs)."""
    ds = platform.sample("sample_index_daily.csv")
    e = platform.run(vol_body(ds, ["random_forest", "gradient_boosting", "lstm", "qml_vqr"]))
    return {"ds": ds, "exp": e, "id": e["experimentId"]}


@pytest.fixture(scope="session")
def opt_run(platform, vol_run):
    ds = platform.sample("sample_option_chain.csv")
    body = {"datasetId": ds["datasetId"], "task": "OPTION_PRICE", "models": ["random_forest", "lstm", "qml_vqr"],
            "seed": 42, "modelConfig": {m: FAST[m] for m in ("lstm", "qml_vqr")}}
    e = platform.run(body)
    return {"ds": ds, "exp": e, "id": e["experimentId"]}
