"""TC-01, TC-02, TC-03, TC-04, TC-05, TC-20, TC-28 (data side) - fast, in-process."""
import io
import math
import statistics
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import SAMPLE

N = 150


def price_csv(n=N, mutate=None):
    d = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(1)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "close": px.round(4)})
    if mutate:
        df = mutate(df)
    return df.to_csv(index=False).encode(), df


def test_tc01_valid_csv_import(inproc):
    raw, df = price_csv()
    r = inproc.upload(raw)
    j = r.json()
    assert r.status_code == 200 and j["status"] == "VALID"
    assert (j["rows"], j["columns"], j["type"]) == (N, 2, "PRICE_SERIES")
    assert (j["dateMin"], j["dateMax"]) == (df["date"].iloc[0], df["date"].iloc[-1])
    assert len(j["datasetHash"]) == 64


def test_tc02_missing_required_column_blocks_everything(inproc):
    raw, _ = price_csv(mutate=lambda d: d.rename(columns={"close": "settle"}))
    r = inproc.upload(raw)
    e = r.json()["error"]
    assert r.status_code == 422 and e["code"] == "SCHEMA_INVALID" and e["fields"] == ["close"]
    assert "close" in e["message"]
    # nothing was stored, so no features/model can be built from it
    assert inproc.ingest.get("/datasets").json()["datasets"] == []


def test_tc03_invalid_values_all_listed_and_blocking(inproc):
    def bad(d):
        d = d.astype(object).copy()
        d.loc[5, "date"] = "not-a-date"
        d.loc[9, "close"] = "-4"
        d.loc[12, "close"] = "abc"
        d.loc[20, "close"] = ""
        d.loc[31, "date"] = d.loc[30, "date"]
        return d
    raw, _ = price_csv(mutate=bad)
    j = inproc.upload(raw).json()
    codes = {i["code"] for i in j["validationReport"]["issues"]}
    assert j["status"] == "BLOCKED"
    assert {"DATE_UNPARSEABLE", "NON_POSITIVE_PRICE", "NOT_NUMERIC", "MISSING_VALUE", "DUPLICATE_DATE"} <= codes
    by = {i["code"]: i for i in j["validationReport"]["issues"]}
    assert 7 in by["DATE_UNPARSEABLE"]["rows"]            # row numbers are file lines (header = 1)
    assert all("not-a-date" not in str(i) for i in j["validationReport"]["issues"])  # no raw values echoed
    r = inproc.build(j["datasetId"])
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_FAILED"


def test_tc04_features_match_hand_calculation(inproc):
    raw, df = price_csv()
    ds = inproc.upload(raw).json()["datasetId"]
    r = inproc.build(ds, horizon=5)
    assert r.status_code == 200
    fs = r.json()
    assert fs["featureSetVersion"].startswith("fv1-") and fs["windows"] == [5, 10, 22]
    prev = inproc.features.get(f"/featuresets/{fs['featureSetId']}/data", params={"split": "train", "limit": 3}).json()
    names = prev["featureNames"]
    close = df["close"].tolist()
    ret = [math.log(close[i] / close[i - 1]) for i in range(1, len(close))]  # ret[i-1] is return at t=i
    first_t = 22 + 3   # first labelled row index (max window + 3 lags)
    for k, ts in enumerate(prev["timestamps"]):
        t = df.index[df["date"] == ts][0]
        assert t == first_t + k
        row = dict(zip(names, prev["X_raw"][k]))
        for w in (5, 10, 22):
            exp = statistics.stdev(ret[t - w:t]) * math.sqrt(252)   # returns t-w+1..t
            assert row[f"rv_{w}"] == pytest.approx(exp, rel=1e-9)
        assert row["ret"] == pytest.approx(ret[t - 1], rel=1e-9)
        assert row["ret_lag1"] == pytest.approx(ret[t - 2], rel=1e-9)
        target = math.sqrt(252 * sum(x * x for x in ret[t:t + 5]) / 5)   # returns t+1..t+5
        assert prev["y"][k] == pytest.approx(target, rel=1e-9)


def test_tc04_option_features_match_hand_calculation(inproc):
    sys_path = str(SAMPLE.parents[1] / "scripts")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", sys_path + "/make_sample_data.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    df = mk.option_chain(n=80)
    ds = inproc.upload(df.to_csv(index=False).encode(), "opts.csv").json()
    assert ds["type"] == "OPTION_CHAIN" and ds["status"] == "VALID"
    fs = inproc.build(ds["datasetId"], task="OPTION_PRICE").json()
    prev = inproc.features.get(f"/featuresets/{fs['featureSetId']}/data", params={"split": "train", "limit": 5}).json()
    for k, ts in enumerate(prev["timestamps"]):
        row = dict(zip(prev["featureNames"], prev["X_raw"][k]))
        cand = df[(df["date"] == ts)]
        match = [c for _, c in cand.iterrows()
                 if abs(math.log(c.underlying_price / c.strike) - row["log_moneyness"]) < 1e-9
                 and abs((pd.Timestamp(c.expiry) - pd.Timestamp(c.date)).days / 365 - row["tte"]) < 1e-9
                 and (c.option_type == "call") == (row["is_call"] > .5)]
        assert match, "feature row must correspond to a source contract"
        assert 0.02 < row["prev_day_iv"] < 3


def test_tc05_chronological_split_and_train_only_scaler(inproc):
    raw, df = price_csv(n=300)
    ds = inproc.upload(raw).json()["datasetId"]
    fs = inproc.build(ds, horizon=5).json()
    r = fs["splitRanges"]
    assert r["train"]["to"] < r["val"]["from"] and r["val"]["to"] < r["test"]["from"]
    assert fs["embargoRows"] == 5
    full = inproc.features.get(f"/featuresets/{fs['featureSetId']}").json()
    tr0, tr1 = full["splits"]["train"]
    allrows = np.array(inproc.features.get(f"/featuresets/{fs['featureSetId']}/data",
                                           params={"split": "train", "limit": 50}).json()["X_raw"])
    from ifap_common.featureset import FeatureSet
    f = FeatureSet.load(fs["featureSetId"])
    train_mean = f.a["X_raw"][tr0:tr1].mean(axis=0)
    assert np.allclose(f.mean, train_mean) and not np.allclose(f.mean, f.a["X_raw"].mean(axis=0))
    assert full["scaler"]["fittedOn"] == "train"
    # rebuilding gives identical ids and hashes (NFR-05)
    again = inproc.build(ds, horizon=5).json()
    assert (again["featureSetId"], again["splitHash"], again["testTimestampHash"]) == \
           (fs["featureSetId"], fs["splitHash"], fs["testTimestampHash"])


@pytest.mark.slow
def test_tc20_validation_performance_50k_rows(inproc):
    n = 50_000
    d = pd.bdate_range("1990-01-02", periods=n)
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.random((n, 20)), columns=[f"x{i}" for i in range(20)])
    df.insert(0, "date", d.strftime("%Y-%m-%d"))
    df["close"] = 100 + rng.random(n)
    raw = df.to_csv(index=False).encode()
    t0 = time.perf_counter()
    r = inproc.upload(raw)
    dt = time.perf_counter() - t0
    print(f"TC-20: {n} rows x {df.shape[1]} cols validated in {dt:.2f}s")
    assert r.status_code == 200 and r.json()["status"] == "VALID" and dt < 10


def test_tc28_upload_rules_and_hostile_cells(inproc):
    assert inproc.upload(b"MZ\x90\x00\x03", "evil.exe").json()["error"]["code"] == "FILE_TYPE_UNSUPPORTED"
    assert inproc.upload(b"PK\x03\x04\x00\x00", "book.xlsx").json()["error"]["code"] == "FILE_TYPE_UNSUPPORTED"
    assert inproc.upload(b"PK\x03\x04\x00\x00data", "renamed.csv").json()["error"]["code"] == "FILE_TYPE_UNSUPPORTED"
    raw, df = price_csv(mutate=lambda d: d.assign(note="=cmd|' /C calc'!A0", **{"<script>alert(1)</script>": "<img src=x onerror=alert(1)>"}))
    j = inproc.upload(raw).json()
    assert j["status"] == "VALID"                       # hostile text is just data in ignored columns
    assert "=cmd" not in str(j) and "onerror" not in str(j)
