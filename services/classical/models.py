"""Classical models: GARCH(1,1) baseline, Black-Scholes baseline, Random Forest, Gradient Boosting (FR-06)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ifap_common.bs import bs_price
from ifap_common.errors import AppError
from ifap_common.modelservice import BaseModelImpl


class Garch(BaseModelImpl):
    name, label, tasks = "garch", "GARCH(1,1) (baseline)", ["VOLATILITY"]
    description = ("Constant-mean GARCH(1,1) fitted by maximum likelihood on TRAIN-period returns only; "
                   "h-day volatility forecast from the fixed-parameter variance recursion.")
    default_config = {"distribution": "normal"}

    def fit(self, fs, config, seed, out_dir: Path):
        from arch import arch_model
        r = fs.a["ret_all"][1:fs.meta["trainReturnEnd"] + 1] * 100.0
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
        p = {k: float(v) for k, v in res.params.items()}
        out = {"mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
               "init_var": float(np.var(r)), "nTrain": int(len(r))}
        (out_dir / "garch.json").write_text(json.dumps(out))
        return {"params": out, "seed": seed, "note": "deterministic (no randomness in MLE start values)"}

    def predict(self, fs, frame, out_dir: Path):
        g = json.loads((out_dir / "garch.json").read_text())
        r = fs.a["ret_all"] * 100.0
        sig2 = np.empty(len(r))  # sigma^2_{t+1} given information up to t
        var = g["init_var"]
        for t in range(len(r)):
            e = (r[t] - g["mu"]) if t > 0 else 0.0
            var = g["omega"] + g["alpha"] * e * e + g["beta"] * var
            sig2[t] = var
        pers = min(g["alpha"] + g["beta"], 0.9999)
        V = g["omega"] / (1 - pers)
        h = fs.horizon
        k = np.arange(h)
        avg = V + (sig2[frame.pos][:, None] - V) * (pers ** k)[None, :]
        return np.sqrt(252 * avg.mean(axis=1) / 1e4)


class BlackScholes(BaseModelImpl):
    name, label, tasks = "black_scholes", "Black-Scholes, trailing vol (baseline)", ["OPTION_PRICE"]
    description = "European Black-Scholes price using the longest trailing realised volatility as sigma."
    default_config = {"vol_feature": "auto"}

    def validate_config(self, config):
        c = super().validate_config(config)
        return c

    def fit(self, fs, config, seed, out_dir: Path):
        vf = config["vol_feature"]
        if vf == "auto":
            vf = f"rv_{max(fs.meta['windows'])}"
        if vf not in fs.names:
            raise AppError("CONFIG_INVALID", "vol_feature is not a feature of this feature set.")
        (out_dir / "bs.json").write_text(json.dumps({"vol_feature": vf}))
        return {"volFeature": vf}

    def predict(self, fs, frame, out_dir: Path):
        vf = json.loads((out_dir / "bs.json").read_text())["vol_feature"]
        raw = frame.X_raw
        c = lambda n: fs.col(raw, n)  # noqa: E731
        S = c("S")
        K = S * np.exp(-c("log_moneyness"))
        return bs_price(S, K, c("tte"), c("rate"), c(vf), c("is_call") > 0.5)


class _SK(BaseModelImpl):
    def make(self, config, seed):  # pragma: no cover
        raise NotImplementedError

    def fit(self, fs, config, seed, out_dir: Path):
        tr0, tr1 = fs.split_range("train")
        raw = fs.a["X_raw"][tr0:tr1]
        X = fs.scale(raw)
        y = fs.a["y"][tr0:tr1] / self.target_scale(fs, raw)
        m = self.make(config, seed)
        m.fit(X, y)
        joblib.dump(m, out_dir / "model.joblib")
        return {"seed": seed, "nTrainRows": int(len(y)), "params": {k: v for k, v in config.items()}}

    def predict(self, fs, frame, out_dir: Path):
        m = joblib.load(out_dir / "model.joblib")  # only after SHA-256 verification by the base service
        return m.predict(frame.X) * self.target_scale(fs, frame.X_raw)


class RandomForest(_SK):
    name, label, tasks = "random_forest", "Random Forest", ["VOLATILITY", "OPTION_PRICE"]
    description = "Random-forest regressor on the shared scaled feature set (train rows only)."
    default_config = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 3}
    bounds = {"n_estimators": (10, 1000), "max_depth": (1, 50), "min_samples_leaf": (1, 100)}

    def make(self, c, seed):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=int(c["n_estimators"]), max_depth=int(c["max_depth"]),
                                     min_samples_leaf=int(c["min_samples_leaf"]), random_state=seed, n_jobs=1)


class GradientBoosting(_SK):
    name, label, tasks = "gradient_boosting", "Gradient Boosting", ["VOLATILITY", "OPTION_PRICE"]
    description = "Gradient-boosted regression trees (scikit-learn) on the shared scaled feature set."
    default_config = {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 3}
    bounds = {"n_estimators": (10, 1000), "learning_rate": (0.001, 1.0), "max_depth": (1, 12)}

    def make(self, c, seed):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=int(c["n_estimators"]), learning_rate=float(c["learning_rate"]),
                                         max_depth=int(c["max_depth"]), random_state=seed)
