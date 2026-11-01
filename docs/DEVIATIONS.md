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
| 10 | TC-24 clean-machine deployment in under 30 minutes | docker compose up --build has now been run on the project owner's machine (all services healthy) and in GitHub Actions (CI run passes, including the compose smoke test). A timed run on a second clean machine is still to be recorded in docs/TEST_RESULTS.md | The build sandbox had no Docker daemon, so this was verified later on real machines. |
| 11 | Sample data | Synthetic only (`scripts/make_sample_data.py`) | Results demonstrate the platform, not real market behaviour. On the volatility sample the ML models win on validation but GARCH wins on test. Selection never uses test data. |
| 12 | Part 2 schedule: backlog and project board created by 12 Oct, Sprint-1 19-23 Oct, Sprint-2 26-30 Oct | The code was developed first. The GitHub repo, backlog (45 issues with story points and assignees), Project board, branch protection and CI were set up on 1 Nov 2026. Issues for work already implemented were closed with a note that they were completed in the initial import | The GitHub process was done late. Dates in the repo are the real dates; nothing was backdated. The Jira project (SM-1..47) was used for planning in the meantime. |
| 13 | Teammate contributions across the sprints | Teammates' pull requests (docs, tests, manual test records, demo videos) were made after the initial code import, from 1 Nov 2026 | The commit history shows the real timeline. The Contributors graph reflects work done on GitHub only. | 
| 14 | Implementation approach | Implemented with AI assistance (permitted by the Part 2 brief); the project owner reviewed, ran and tested the result | Stated for transparency. |
