# Deviations from the Part 1 specification

This list is kept so the submission is honest about where the implementation differs from, or goes beyond, the Part 1 document.

| # | Spec / expectation | What was built | Why / impact |
|---|---|---|---|
| 1 | Message queue (Redis/RabbitMQ) for asynchronous dispatch | In-process job queue inside the orchestrator | Fewer moving parts for a mini-project. Experiment jobs survive neither an orchestrator restart nor run on multiple orchestrator replicas. Per-model timeouts and failure isolation are unchanged. |
| 2 | Dashboard as a separate component | Static dashboard served by the gateway | One fewer container; same-origin API calls; CSP applied by the gateway. |
| 3 | Implied volatility as a feature for option prices | IV is computed and recorded but **excluded from option-price model inputs**; `prev_day_iv` is used instead | Same-day IV is derived from the option price, so using it would leak the target. |
| 4 | Prediction for any horizon up to 30 days | The requested horizon must equal the champion's trained horizon | Models are trained for one horizon; other horizons would need a retrain. |
| 5 | Directional accuracy for every task | DA is reported as n/a for option prices | A direction of an option price level is not meaningful for this task. |
| 6 | LSTM early stopping | Uses the last 15 % of **training** rows as the stopping set | Validation rows stay reserved for model selection. |
| 7 | QML input features | Uses the first `n_qubits` scaled features and a seeded training subsample of at most 400 rows | Keeps simulator time bounded; results therefore reflect a deliberately small quantum model. |
| 8 | Upload limit | 25 MB hard cap (matches FR-01) | No deviation in value; noted because it is enforced in the gateway and ingestion. |
| 9 | TC-25 usability (3 first-time users) and Firefox check | Manual; not automated | Needs real people and a second browser. Chrome at 1366x768 was checked by script. |
| 10 | TC-24 clean-machine deployment in under 30 minutes | `docker-compose.yml` is statically tested and passes `docker compose config`, but **`docker compose up` has not been executed** | No Docker daemon was available in the build sandbox. CI runs the full compose smoke test (`.github/workflows/ci.yml`); the clean-machine timing must be recorded manually. |
| 11 | Sample data | Synthetic only (`scripts/make_sample_data.py`) | Results demonstrate the platform, not real market behaviour. On the volatility sample the ML models win on validation but GARCH wins on test. Selection never uses test data. |
