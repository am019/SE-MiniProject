"""Generate the SYNTHETIC sample datasets (clearly labelled; not real market data).

    python scripts/make_sample_data.py [outdir]
Deterministic (fixed seeds) so every machine produces identical files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from ifap_common.bs import bs_price  # noqa: E402


def simulate(n, seed, omega=2e-6, alpha=0.06, beta=0.89, gamma=0.06, mu=0.0003):
    rng = np.random.default_rng(seed)
    h = omega / (1 - alpha - beta - gamma / 2)
    r = np.zeros(n)
    sig = np.zeros(n)
    for t in range(n):
        sig[t] = np.sqrt(h)
        e = rng.standard_t(8) / np.sqrt(8 / 6)
        r[t] = mu + sig[t] * e
        h = omega + (alpha + gamma * (r[t] < mu)) * (r[t] - mu) ** 2 + beta * h
    return r, sig


def price_series(n=1500, seed=7, start="2018-01-01", s0=100.0):
    r, sig = simulate(n, seed)
    close = s0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close.round(4),
                         "volume": (1e6 * (1 + 30 * np.abs(r)) * np.random.default_rng(seed + 1).uniform(.8, 1.2, n)).astype(int)})


def option_chain(n=320, seed=11, start="2022-01-03", s0=4000.0, rate=0.02):
    r, sig = simulate(n, seed, omega=3e-6)
    S = s0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed + 5)
    rows = []
    for i, d in enumerate(dates):
        # "true" vol the market prices off: forward-looking blend of current and mean-reverting level
        base = sig[min(i + 3, n - 1)] * np.sqrt(252)
        for days in (30, 60):
            for m in (0.92, 0.96, 1.0, 1.04, 1.08):
                K = round(S[i] * m / 5) * 5
                lm = np.log(S[i] / K)
                vol = base * (1 + 0.5 * (-lm)) * (1 + 0.04 * (days == 60)) * (1 + rng.normal(0, 0.01))
                for typ in ("call", "put"):
                    p = float(bs_price(S[i], K, days / 365, rate, vol, typ == "call"))
                    rows.append((d.strftime("%Y-%m-%d"), K, (d + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                                 typ, round(S[i], 2), round(max(p, 0.01), 2), rate))
    return pd.DataFrame(rows, columns=["date", "strike", "expiry", "option_type", "underlying_price",
                                       "option_price", "rate"])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "sample")
    out.mkdir(parents=True, exist_ok=True)
    price_series().to_csv(out / "sample_index_daily.csv", index=False)
    option_chain().to_csv(out / "sample_option_chain.csv", index=False)
    print("wrote", [p.name for p in out.iterdir()])
