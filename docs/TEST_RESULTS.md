# Test results

Tester: Aryan Arora (PES1UG24AM055)
Date: 3 November 2026 (all times IST)
Branch: `feature/43-test-results`
Issues: #43 (TASK-QA2), #29 (NFR-09), #30 (NFR-10)

## Test environment

| Item | Value |
|---|---|
| Machine | Acer laptop, Windows |
| Python | 3.14 |
| Docker Compose | v5.4.0 (Docker Desktop, WSL2 backend) |
| Docker resources, default | 32 CPUs visible, 7.609 GiB memory |
| Docker resources, after workaround | limited to 4 processors and 6 GB via `.wslconfig` (see D-01) |
| Browsers | Chrome (manual), Microsoft Edge (usability test, TC-25), Chromium and Firefox (Playwright) |

## Summary

| Test | Result |
|---|---|
| TC-24 Docker start (NFR-07) | Pass: 3 min (cached rebuild), about 11 min or less (first build) against the 30 min target |
| Smoke test through the gateway | Pass on the second configuration (4 CPUs); failed on the default configuration (see D-01) |
| TC-19 Playwright UI test, Chromium and Firefox, 1366x768, no horizontal scroll | Pass (2 passed in 106.98 s) |
| TC-25 / NFR-09 three first-time users | Pass: 3 of 3 finished within 15 minutes (1.0, 0.7, 0.4 min) |
| NFR-10 manual Firefox check and chart readability | Pass: Firefox 160, all flows worked; chart readable |

## TC-24 Clean-machine deployment (NFR-07)

Target: all services healthy in under 30 minutes after `docker compose up --build`.

| Run | Start | Ready | Total | Notes |
|---|---|---|---|---|
| First build (images built from scratch) | between 14:12 and 14:20 (exact time not recorded, no stopwatch) | about 14:23 | about 11 min or less | The build was observed in progress at 14:20. `docker compose ps` showed every service `(healthy)`; the sign-in page loaded. |
| Rebuild with Docker's layer cache | 14:30 | 14:33 | 3 min | Not a clean-machine time, because Docker reused cached layers. |

Result: pass for both runs. The first-build time is an estimate because no stopwatch was running.

## Smoke test (`scripts/smoke.py`)

| Attempt | Configuration | Outcome |
|---|---|---|
| 1 | Default Docker (32 CPUs) | Login rejected because the password was mistyped (user error, not a product defect). |
| 2 | Default Docker (32 CPUs) | `garch` FAILED (`MODEL_FAILED: Model training process terminated unexpectedly.`) and `qml_vqr` FAILED (`MODEL_FAILED: Model training failed.`). `lstm` COMPLETED. The run was interrupted by hand. See D-01. |
| 3 | Default Docker (32 CPUs) | `409 Conflict` on `POST /api/experiments`, because experiment `exp-0001` from the interrupted run was still RUNNING. This matches the intended `EXPERIMENT_IN_PROGRESS` behaviour (one experiment at a time). |
| 4 | Docker limited to 4 processors, 6 GB | Pass. See the results below. |

Results of the passing run:

| Model | Family | Status | Validation RMSE | Test RMSE |
|---|---|---|---|---|
| qml_vqr | qml | COMPLETED (champion) | 0.0521094 | 0.11592 |
| garch (baseline) | classical | COMPLETED | 0.0543 | 0.08912 |
| gradient_boosting | classical | COMPLETED | 0.05496 | 0.13467 |

- Selection: `qml_vqr` had the lowest validation RMSE among 3 eligible models, against baseline `garch` (0.0542971). The selection text states that test-period metrics did not influence the choice.
- Prediction: HTTP 200, model `qml_vqr`, value 0.11729743049249246 (annualised volatility, decimal), as of 2023-09-29.
- Reports: JSON 200, CSV 200, PDF 200.
- Duration of the passing run: not recorded.
- The `qml_vqr` test RMSE (0.11592) was not recorded in the first passing run. It was taken from a second passing smoke run on 14 November 2026 (same configuration: Docker limited to 4 processors, 6 GB). That run again selected `qml_vqr` as champion (validation RMSE 0.05211) and returned HTTP 200 for the prediction and the JSON, CSV and PDF reports.

## TC-19 / FR-18 / NFR-10 Playwright UI test (`tests/test_ui.py`, issue #42)

Command: `python -m pytest tests/test_ui.py -m integration -q -s`

| Browser | Viewport | Flow | Result |
|---|---|---|---|
| Chromium | 1366x768 | login, load sample, run experiment, leaderboard and champion, predict, no horizontal scroll on Overview and Results | Pass |
| Firefox | 1366x768 | same | Pass |

Total: 2 passed in 106.98 s.

Fixture note: the test starts the platform without Docker and uses only the classical models (garch baseline, random forest, gradient boosting) for speed.

## Unit and repository tests

Command: `python -m pytest -m "not integration and not slow" -q`

Result: 28 passed, 28 deselected, 1 failed. The failure is D-02.

## TC-25 / NFR-09 Usability, three first-time users

Target: 3 of 3 users finish in 15 minutes. Task: sign in as `analyst`, load `sample_index_daily.csv`, run an experiment, say which model won, download a report.

Tested on 14 November 2026 in Microsoft Edge. Each tester used the same laptop, one after another, and was timed from clicking the username box until the report downloaded. Testers are identified by initials.

| User | Completed? | Time (min) | Where they hesitated | Comments |
|---|---|---|---|---|
| 1 (A.A.) | Yes (report exp-0005 downloaded) | 1.0 (stopwatch 01:00.81) | None; did not hesitate | Found it easy to do |
| 2 (G.K.) | Yes (report exp-0006 downloaded) | 0.7 (stopwatch 00:40.13) | None; did not hesitate | Found it easy to do |
| 3 (A.R.) | Yes (report exp-0007 downloaded) | 0.4 (stopwatch 00:25.92) | None; did not hesitate | Found it easy to do |

Result: 3 of 3 users completed the task within the 15-minute target. Whether each tester named the winning model correctly was not recorded (all three reports show `qml_vqr` as champion). No ease score (1 to 5) was collected; all three said it was easy. Most common point of confusion: none reported.

## NFR-10 Browser compatibility and chart readability (manual)

Manual check done on 14 November 2026 on Windows (same laptop as above), logged in as `analyst`. The automated Firefox run above also passed. Chart readability was judged by the tester (Aryan Arora). The browser console was not checked for errors.

| Browser / version | Upload | Run | Results | Predict | Export | Chart readable | Notes |
|---|---|---|---|---|---|---|---|
| Firefox 160 | Yes (sample dataset loaded, status VALID) | Yes (experiment completed) | Yes (leaderboard, champion card and chart shown) | Yes (prediction 0.1173 from qml_vqr) | Yes (report downloaded) | Yes | Layout displayed correctly in Firefox; no problems seen |

## Defects found

| # | Description | Severity | Status |
|---|---|---|---|
| D-01 | In the Docker stack on a host that exposes many CPUs (32 CPUs, 7.609 GiB), GARCH and the QML model fail to train. The classical container logs `OpenBLAS error: Memory allocation still failed after 10 retries, giving up.` The platform reports `MODEL_FAILED: Model training process terminated unexpectedly.` (garch) and `MODEL_FAILED: Model training failed.` (qml_vqr). The same models trained fine without Docker (Playwright run) and in Docker after limiting WSL2 to 4 processors. Suggested fix for the owner: set `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` (for example 2) for the classical and qml services in `docker-compose.yml`, or limit library threads in the training code. | High (can break the live demo on many-core machines) | Open, to be reported to the owner (am019) |
| D-02 | `tests/test_repo.py::test_tc24_compose_fails_without_secrets` fails on Windows. The test runs `docker compose` with an environment containing only `PATH`, so Docker cannot find its Compose plugin and prints a generic help text instead of the expected "init_env" message (exit code 125). It passes on Linux CI. | Low (test environment issue, not an app defect) | Open |

## Not run

- A clean-machine `docker compose up --build` on a second computer.
- QML and LSTM timing runs (`pytest tests/test_isolated.py -k "tc21 or tc22_dl"`).
