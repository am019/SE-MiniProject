"""End-to-end smoke test over the public API (login -> sample data -> experiment -> leaderboard -> predict -> report).

    python scripts/smoke.py --base http://127.0.0.1:8080 --user admin --password <pw> [--task VOLATILITY]
"""
import argparse
import sys
import time

import httpx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--task", default="VOLATILITY", choices=["VOLATILITY", "OPTION_PRICE"])
    ap.add_argument("--models", default="random_forest,gradient_boosting,lstm,qml_vqr")
    a = ap.parse_args(argv)
    c = httpx.Client(base_url=a.base, timeout=60)
    tok = c.post("/api/auth/login", json={"username": a.user, "password": a.password}).json()["token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    sample = "sample_index_daily.csv" if a.task == "VOLATILITY" else "sample_option_chain.csv"
    ds = c.post(f"/api/samples/{sample}/load").json()
    print("dataset:", ds["datasetId"], ds["type"], ds["status"], ds["rows"], "rows")
    r = c.post("/api/experiments", json={"datasetId": ds["datasetId"], "task": a.task, "horizon": 5,
                                         "models": a.models.split(","), "seed": 42})
    r.raise_for_status()
    eid = r.json()["experimentId"]
    t0 = time.time()
    while True:
        e = c.get(f"/api/experiments/{eid}").json()
        print(f"  [{int(time.time() - t0):4d}s] {e['status']:9s}", {k: v["status"] for k, v in e["models"].items()})
        if e["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(3)
    if e["status"] != "COMPLETED":
        print("experiment failed:", e.get("error"))
        return 1
    lb = c.get(f"/api/experiments/{eid}/leaderboard").json()
    pm = lb["primaryMetric"]
    print(f"\nLeaderboard (validation {pm} decides; test shown for information)")
    for r in lb["leaderboard"]:
        v = r.get("val", {}).get(pm)
        t = r.get("test", {}).get(pm)
        print(f"  {r.get('rank') or '-':>2} {r['model']:18s} {r['family']:9s} {r['status']:10s} "
              f"val {v if v is None else round(v, 5)!s:>9} test {t if t is None else round(t, 5)!s:>9} "
              f"{'CHAMPION' if r.get('isChampion') else ''}")
    print("\n" + c.get(f"/api/experiments/{eid}/selection").json()["justification"])
    body = {"task": a.task}
    if a.task == "OPTION_PRICE":
        body |= {"strike": 4000, "daysToExpiry": 30, "optionType": "call"}
    p = c.post("/api/predictions", json=body)
    print("\nprediction:", p.status_code, {k: p.json().get(k) for k in ("model", "family", "values", "unit", "asOf")})
    for fmt in ("json", "csv", "pdf"):
        print("report", fmt, c.get(f"/api/reports/{eid}", params={"format": fmt}).status_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
