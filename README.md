# IFAP — Intelligent Financial Analytics Platform

A microservice platform that compares **classical ML**, **deep learning** and **quantum ML (simulator)** for
financial **volatility** and **option-price** prediction, and automatically selects the best model using
**validation data only** (test data is never used for selection).

Spec: `docs/Intelligent_Financial_Analytics_Platform_Part1_v2.docx` (FR-01..20, NFR-01..11, SR-01..08, TC-01..32).
Jira: project SE-MINI (key `SM`). Deviations from the spec: [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## Quick start (Docker)

```bash
python scripts/init_env.py          # writes .env with random secrets and prints the admin/analyst passwords
docker compose up --build           # first build downloads PyTorch (CPU) and PennyLane: allow several minutes
```

Open <http://127.0.0.1:8080> (change with `IFAP_PORT` in `.env`). Only the gateway is published, and only on loopback (SR-02).
Log in as `admin` or `analyst` with the passwords printed by `init_env.py` (also stored in `.env`, which is git-ignored).

End-to-end check against the running stack:

```bash
python scripts/smoke.py --base http://127.0.0.1:8080 --password <admin password from .env>
```

## Quick start (no Docker)

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/dev.txt
python scripts/run_local.py --port 8080     # starts every service as a local process; prints the passwords; data in ./var
python scripts/smoke.py --password <admin password>     # about 80 s
```

## Using the dashboard

1. **Datasets** — upload a UTF-8 `.csv` (max 25 MB) or load a bundled synthetic sample from `data/sample/`.
   Price series need `date, close`; option chains need `date, strike, expiry, option_type, underlying_price, option_price` (the validation report shown after upload lists any problems).
2. **Run experiment** — pick dataset, task (VOLATILITY or OPTION_PRICE), horizon, models and primary metric.
3. **Results** — leaderboard, selection justification and chart; export a PDF/CSV/JSON report.
4. **Predict** — forecast with the current champion (horizon must equal the champion's trained horizon).
5. **Registry** — runs, hashes, champion history and roll-back. **Admin** — service health and configuration.

> Sample data is synthetic. On the volatility sample the ML models win on validation while GARCH wins on test.
> That is an honest outcome, not a bug: selection never looks at test data.

## Architecture

| Service | Role |
|---|---|
| `gateway` | only public entry point: JWT auth, RBAC, rate limit (60 req/min/user), CSP, serves the dashboard |
| `ingestion` | upload, validation, dataset hashing |
| `features` | log returns, realised volatility, chronological split, shared feature set |
| `orchestrator` | experiment jobs, per-model timeouts, failure isolation |
| `classical` / `dl` / `qml` | model services (GARCH, Black-Scholes, Random Forest, Gradient Boosting / LSTM / variational quantum regressor, ≤ 8 qubits) |
| `evaluation` | metrics on identical test timestamps, leaderboard, selection policy |
| `registry` | runs, artifact SHA-256 hashes, champion pointer, history, roll-back |
| `report` | PDF / CSV / JSON export with provenance |
| `stub` | optional extensibility demo (NFR-06) |

All internal services sit on a private Docker network and require a service token. Every model service implements the same
contract (`train`, `predict`, `describe`, `health`), implemented once in `common/ifap_common/modelservice.py`.

## API summary (all via the gateway; JWT bearer token required except login and `/health`)

| Method & path | Purpose |
|---|---|
| `POST /api/auth/login` | obtain a token |
| `GET /api/me` | current user and role |
| `POST /api/datasets`, `GET /api/datasets[/{id}[/summary]]` | upload / list / inspect datasets |
| `GET /api/samples`, `POST /api/samples/{name}/load` | bundled sample data |
| `GET /api/models/available` | models offered by the configuration |
| `POST /api/experiments`, `GET /api/experiments[/{id}[/leaderboard, /selection]]` | run and inspect experiments |
| `POST /api/predictions` | prediction with the champion |
| `GET /api/models[/{id}]`, `DELETE /api/models/{id}` | registry |
| `GET /api/models/champion[s]`, `GET /api/models/champion/history`, `POST /api/models/champion/rollback` | champion management |
| `GET /api/reports/{experiment_id}?format=pdf\|csv\|json` | report export |
| `GET /api/admin/health`, `GET|PUT /api/admin/config` | **Administrator only** |

## Adding a model (NFR-06)

1. Implement a service that follows the common contract (copy `services/stub` as a starting point).
2. Add its URL under `services` and its models under `models` in `config/models.json` (see `config/models.stub.json`).
3. In Docker, add one service block to `docker-compose.yml`. No change is needed in the gateway, orchestrator, evaluator or dashboard.

Extensibility demo: `IFAP_MODELS_CONFIG=/app/config/models.stub.json docker compose --profile stub up --build`.

## Configuration

Secrets come only from environment variables (SR-04): `IFAP_JWT_SECRET`, `IFAP_SERVICE_TOKEN`, `IFAP_ADMIN_PASSWORD`,
`IFAP_ANALYST_PASSWORD`. Other settings: `IFAP_PORT`, `IFAP_MODELS_CONFIG`, `IFAP_MAX_UPLOAD_MB` (25), `IFAP_MAX_QUBITS` (8),
`IFAP_JOB_MEM_MB`, `IFAP_RATE_LIMIT`, `IFAP_JWT_TTL_MIN`. Model-service containers have CPU, memory and PID limits (SR-07).

## Tests

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements/dev.txt
pytest -m "not integration and not slow" -q     # unit + repository checks (fast)
pytest -m integration -q                         # real multi-process platform (several minutes)
pytest tests/test_isolated.py -k "tc21 or tc22_dl" -s    # slow timing / reproducibility tests; prints REPRODUCIBILITY numbers
```

Manual test cases: **TC-24** timing of a clean-machine `docker compose up`, and **TC-25** usability with three first-time users plus a Firefox check.

## Traceability

| Requirement | Jira | Implemented in | Test case(s) | Automated in |
|---|---|---|---|---|
| FR-01 | SM-9 | services/ingestion, gateway | TC-01 | test_unit_data |
| FR-02 | SM-10 | services/ingestion, common/ifap_common/datasets.py | TC-02, TC-03 | test_unit_data |
| FR-03 | SM-11 | services/ingestion, services/features | TC-02, TC-03 | test_unit_data |
| FR-04 | SM-12 | services/features/build.py | TC-04, TC-14 | test_unit_data, test_integration |
| FR-05 | SM-13 | services/features/build.py, common/ifap_common/featureset.py | TC-05 | test_unit_data |
| FR-06 | SM-14 | services/classical, common/ifap_common/modelservice.py | TC-06 | test_integration |
| FR-07 | SM-15 | services/dl | TC-07 | test_integration |
| FR-08 | SM-16 | services/qml | TC-08 | test_integration |
| FR-09 | SM-17 | common/ifap_common/modelservice.py (common contract) | TC-06, TC-07, TC-08 | test_integration |
| FR-10 | SM-18 | services/orchestrator, gateway, web/ | TC-09, TC-14 | test_integration |
| FR-11 | SM-19 | services/evaluation/metrics.py | TC-10, TC-14 | test_unit_selection, test_integration |
| FR-12 | SM-20 | services/evaluation, web/app.js (results) | TC-11 | test_integration |
| FR-13 | SM-21 | services/evaluation (selection policy) | TC-12, TC-13 | test_unit_selection, test_integration |
| FR-14 | SM-22 | services/gateway, registry, model services | TC-15 | test_integration |
| FR-15 | SM-23 | services/orchestrator, common/ifap_common/isolated.py (timeout/memory) | TC-13, TC-16 | test_unit_selection, test_isolated |
| FR-16 | SM-24 | services/registry | TC-17 | test_isolated |
| FR-17 | SM-25 | services/report | TC-18 | test_integration |
| FR-18 | SM-26 | web/index.html, web/app.js | TC-19 | test_integration |
| FR-19 | SM-27 | services/registry, orchestrator | TC-17 | test_isolated |
| FR-20 | SM-28 | services/gateway (/api/admin/health), web/app.js | TC-19 | test_integration |
| NFR-01 | SM-29 | services/ingestion (vectorised validation) | TC-20 | test_unit_data |
| NFR-02 | SM-30 | services/gateway, registry | TC-15 | test_integration |
| NFR-03 | SM-31 | services/classical, dl, qml | TC-21 | test_isolated (slow) |
| NFR-04 | SM-32 | services/orchestrator | TC-16 | test_isolated |
| NFR-05 | SM-33 | services/features, seeds in common | TC-22 | test_isolated, test_unit_data |
| NFR-06 | SM-34 | config/models.json, services/stub | TC-23 | test_isolated |
| NFR-07 | SM-35 | Dockerfile, docker-compose.yml, requirements/ | TC-24 | test_repo (static); clean-machine run is manual |
| NFR-08 | SM-36 | common/ifap_common/appfactory.py (JSON logs, request IDs) | TC-19 | test_integration |
| NFR-09 | SM-37 | web/ | TC-25 | manual (3 users) |
| NFR-10 | SM-38 | web/style.css | TC-25 | test_repo (contrast); Firefox check is manual |
| NFR-11 | SM-39 | services/evaluation, services/features | TC-10 | test_unit_selection, test_integration |
| SR-01 | SM-40 | services/gateway (JWT) | TC-26 | test_integration |
| SR-02 | SM-41 | docker-compose.yml (networks, ports) | TC-27 | test_integration, test_repo |
| SR-03 | SM-42 | services/ingestion | TC-28 | test_unit_data, test_integration |
| SR-04 | SM-43 | scripts/init_env.py, .gitignore, .env.example | TC-29 | test_repo |
| SR-05 | SM-44 | services/registry, common model-service base (SHA-256) | TC-18, TC-30 | test_integration, test_isolated |
| SR-06 | SM-45 | common/ifap_common/errors.py | TC-31 | test_integration, test_isolated |
| SR-07 | SM-46 | common/ifap_common/isolated.py, gateway rate limit, compose limits | TC-21, TC-32 | test_isolated, test_repo |
| SR-08 | SM-47 | services/gateway (RBAC) | TC-26 | test_integration |

## Repository layout

```
common/ifap_common/   shared library (config, app factory, clients, isolated training, model-service base)
services/<name>/      one directory per microservice (also one container per service)
web/                  dashboard (static HTML/CSS/JS served by the gateway)
config/               models.json (+ models.stub.json)
data/sample/          synthetic sample CSVs (scripts/make_sample_data.py)
scripts/              init_env.py, run_local.py, smoke.py, make_sample_data.py
tests/                unit, integration, isolated-platform and repository tests
```
