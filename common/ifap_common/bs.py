"""Vectorised Black-Scholes pricing and implied-volatility inversion (European options)."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def bs_price(S, K, T, r, sigma, is_call):
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    disc = np.exp(-r * T)
    call = S * ndtr(d1) - K * disc * ndtr(d2)
    put = K * disc * ndtr(-d2) - S * ndtr(-d1)
    return np.where(is_call, call, put)


def implied_vol(price, S, K, T, r, is_call, lo=1e-4, hi=5.0, iters=60):
    """Bisection (monotone in sigma). Returns NaN where the price violates no-arbitrage bounds."""
    price, S, K, T, r = (np.asarray(x, dtype=float) for x in (price, S, K, T, r))
    is_call = np.asarray(is_call, dtype=bool)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        pm = bs_price(S, K, T, r, m, is_call)
        up = pm < price
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    iv = 0.5 * (a + b)
    plo, phi = bs_price(S, K, T, r, lo, is_call), bs_price(S, K, T, r, hi, is_call)
    bad = (price < plo - 1e-12) | (price > phi + 1e-12) | ~np.isfinite(price)
    return np.where(bad, np.nan, iv)
