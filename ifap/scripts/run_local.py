"""Run the whole platform as local processes (no Docker) - used for development and tests.

    python scripts/run_local.py [--data-dir var] [--port 8080] [--models-config config/models.json]

Starts one uvicorn process per service (internal ones on 127.0.0.1 only, gateway on --port).
Per-service logs are written to <data-dir>/logs/<service>.log (JSON lines with correlation IDs).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = ["registry", "ingestion", "features", "evaluation", "report", "orchestrator"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "var"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--base-port", type=int, default=9100)
    ap.add_argument("--models-config", default=str(ROOT / "config" / "models.json"))
    ap.add_argument("--only", default="", help="comma list of services to start (testing fault isolation)")
    a = ap.parse_args(argv)
    data = Path(a.data_dir).resolve()
    (data / "logs").mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.models_config).read_text())
    extra = list(cfg["services"].keys())
    names = BASE + extra
    ports = {n: a.base_port + i for i, n in enumerate(names)}
    env = dict(os.environ, IFAP_DATA_DIR=str(data), IFAP_MODELS_CONFIG=str(Path(a.models_config).resolve()),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "common")]), PYTHONUNBUFFERED="1")
    for n, p in ports.items():
        env[f"IFAP_URL_{n.upper()}"] = f"http://127.0.0.1:{p}"
    sys.path[:0] = [str(ROOT / "common"), str(ROOT)]
    os.environ.update(IFAP_DATA_DIR=str(data))
    from ifap_common.config import get_secret  # creates .secrets.json once, before children race for it
    get_secret("IFAP_JWT_SECRET")
    only = {s for s in a.only.split(",") if s}
    procs = {}

    def start(name, module, port):
        log = open(data / "logs" / f"{name}.log", "a")
        procs[name] = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"{module}:app", "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "warning"], cwd=str(ROOT), env=env, stdout=log, stderr=log)

    for n in names:
        if not only or n in only:
            start(n, f"services.{n}.main", ports[n])
    start("gateway", "services.gateway.main", a.port)

    def stop(*_):
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    t0 = time.time()
    pending = {n: f"http://127.0.0.1:{ports.get(n, a.port)}/health" for n in procs}
    while pending and time.time() - t0 < 120:
        for n, u in list(pending.items()):
            try:
                if httpx.get(u, timeout=1).status_code == 200:
                    del pending[n]
            except httpx.HTTPError:
                pass
        time.sleep(0.3)
    if pending:
        print("services failed to start:", ", ".join(pending), "- see", data / "logs")
        stop()
    sec = json.loads((data / ".secrets.json").read_text())
    print(f"\nIFAP is running:  http://127.0.0.1:{a.port}\n"
          f"  Administrator:  admin   / {sec['IFAP_ADMIN_PASSWORD']}\n"
          f"  Analyst:        analyst / {sec['IFAP_ANALYST_PASSWORD']}\n"
          f"  Logs: {data / 'logs'}   (Ctrl+C to stop)\n", flush=True)
    while True:
        time.sleep(1)
        for n, p in procs.items():
            if p.poll() is not None and n not in ("__never__",):
                pass  # a crashed service is shown as DOWN on the Health page; others keep running


if __name__ == "__main__":
    sys.exit(main())
