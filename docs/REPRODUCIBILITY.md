# Reproducibility and timing evidence (NFR-05 / TC-22, NFR-03 / TC-21)

Measured on the build sandbox (Python 3.13, CPU only, torch 2.14.1, PennyLane 0.45.1) with
`pytest tests/test_isolated.py -k "tc21 or tc22_dl" -s`. Both tests passed (2 passed, 123 s).
Sample data is synthetic, volatility task, horizon 5.

## DL / QML run-to-run variation (validation RMSE, 3 runs each)

| Model | Same seed (42, 42, 42) | Different seeds (1, 2, 3) |
|---|---|---|
| LSTM | 0.052741, 0.052741, 0.052741 (identical, spread 0) | 0.052509, 0.052589, 0.052856 (spread 0.000347) |
| QML (variational regressor) | 0.053011, 0.053011, 0.053011 (identical, spread 0) | 0.053822, 0.053385, 0.053201 (spread 0.000622) |

With the same seed, data and configuration the results were bit-identical on this machine. Different seeds change validation RMSE by
about 0.7 % (LSTM) and 1.2 % (QML) of its value. Identical results are only expected on the same hardware and library versions; the test therefore asserts a
tolerance (< 0.02), not exact equality. Classical models and the split-timestamp hash are checked exactly by TC-22 (classical) and TC-05.

## Training time on a 5,000-row price series (limit: classical 300 s, DL 900 s)

| Model | Train time (s) |
|---|---|
| GARCH(1,1) | 1.6 |
| Gradient Boosting | 3.1 |
| Random Forest | 6.2 |
| LSTM | 6.9 |

All are far below the NFR-03 bounds. The QML model was not part of this 5,000-row timing run (it trains on a seeded subsample of at most 400 rows);
its time on the standard sample is visible in the leaderboard of any experiment. The same test also confirms that a hanging job is stopped at its
timeout and a memory-hungry job is stopped, while the rest of the platform stays up (SR-07).
