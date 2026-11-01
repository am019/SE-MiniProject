"""TC-10, TC-12, TC-13 - metrics and the selection policy, no services needed."""
import math

import pytest

from services.evaluation import metrics as M
from services.evaluation.main import ModelResult as R, SelectReq, select


def test_tc10_metrics_match_hand_calculation():
    m = M.compute("VOLATILITY", [0.2, 0.3], [0.25, 0.2])
    assert m["RMSE"] == pytest.approx(math.sqrt((0.05 ** 2 + 0.1 ** 2) / 2))
    assert m["MAE"] == pytest.approx(0.075)
    q = lambda y, h: (y * y) / (h * h) - math.log((y * y) / (h * h)) - 1  # noqa: E731
    assert m["QLIKE"] == pytest.approx((q(0.2, 0.25) + q(0.3, 0.2)) / 2)
    assert m["MAPE"] == pytest.approx((25.0 + 100 / 3) / 2)
    assert m["DA"] == pytest.approx(1.0) or m["DA"] == 0.0   # one change: predicted down, actual up -> 0
    assert m["DA"] == 0.0


def test_tc10_mape_na_when_actual_zero_and_da():
    m = M.compute("VOLATILITY", [0.2, 0.0, 0.3], [0.25, 0.1, 0.2])
    assert m["MAPE"] is None
    assert m["DA"] == 1.0     # both consecutive changes have the same sign
    o = M.compute("OPTION_PRICE", [1.0, 2.0], [1.1, 1.8])
    assert o["MAPE"] == pytest.approx(10.0) and "QLIKE" not in o and o["DA"] is None


def results(**kw):
    base = dict(garch=(.100, 5), qml=(.060, 2), dl=(.050, 9), rf=(.070, 4))
    base.update(kw)
    fam = dict(garch="classical", qml="qml", dl="dl", rf="classical")
    return [R(model=k, family=fam[k], status="COMPLETED", val={"RMSE": v}, inferenceTimeMs=t,
              test={"RMSE": {"qml": .01, "dl": .09}.get(k, .08)}) for k, (v, t) in base.items()]


def test_tc12_champion_is_validation_best_not_test_best():
    out = select(SelectReq(baseline="garch", results=results()))
    assert out["selection"]["champion"]["model"] == "dl"          # QML is best on TEST (0.01) but not chosen
    assert out["selection"]["policy"]["selectionData"] == "validation only"
    assert "did not influence" in out["selection"]["justification"]
    assert [r["model"] for r in out["leaderboard"]][:2] == ["dl", "qml"]


def test_tc12_near_tie_prefers_lower_inference_time():
    out = select(SelectReq(baseline="garch", results=results(dl=(.0500, 9), qml=(.0504, 1))))
    assert out["selection"]["champion"]["model"] == "qml"
    assert "tie" in out["selection"]["justification"]


def test_tc12_margin_requirement():
    out = select(SelectReq(baseline="garch", margin=0.45, results=results()))   # needs <= 0.055
    assert out["selection"]["champion"]["model"] == "dl" and out["selection"]["flags"] == []
    out = select(SelectReq(baseline="garch", margin=0.5, results=results()))    # needs < 0.050: nobody
    assert out["selection"]["champion"]["model"] == "garch"
    assert "NO_MODEL_BEAT_BASELINE" in out["selection"]["flags"]


def test_tc13_baseline_fallback_and_failed_model_excluded():
    rs = results(dl=(.2, 9), qml=(.3, 2), rf=(.15, 4))
    rs.append(R(model="lstm2", family="dl", status="TIMED_OUT", reason="TIMEOUT"))
    out = select(SelectReq(baseline="garch", results=rs))
    s = out["selection"]
    assert s["champion"]["model"] == "garch" and s["flags"] == ["NO_MODEL_BEAT_BASELINE"]
    assert s["excluded"] == [{"model": "lstm2", "reason": "TIMEOUT", "status": "TIMED_OUT"}]
    assert any(r["model"] == "lstm2" and r["rank"] is None for r in out["leaderboard"])
    assert "no model beat the baseline" in s["justification"].lower()
