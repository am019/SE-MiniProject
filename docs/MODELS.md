# Model Services Documentation

## 1. Overview

The Intelligent Financial Analytics Platform provides classical machine-learning models, a deep-learning model, and a variational quantum machine-learning model for financial volatility forecasting and option pricing.

The model services share a common API contract implemented by `ifap_common.modelservice`. Each service exposes training, prediction, description, and health endpoints.

Supported model implementations include:

- GARCH(1,1)
- Black–Scholes
- Random Forest
- Gradient Boosting
- Long Short-Term Memory (LSTM)
- Variational Quantum Regressor (QML)

## 2. Common Model-Service Contract

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/train` | Train a model using a feature set and configuration |
| POST | `/predict` | Generate predictions using a trained model |
| GET | `/describe` | Return model information, configurations, library versions, and service limits |
| GET | `/health` | Check service health |

### Training inputs

The training request accepts:

- `featureSetId`: Identifier of the feature set.
- `model`: Registered model name.
- `config`: Optional model-specific configuration.
- `seed`: Random seed; defaults to `42`.
- `timeoutSec`: Training timeout in seconds; defaults to `1200`.

### Prediction inputs

The prediction request accepts:

- `modelId`: Identifier of the trained model.
- `featureSetId`: Identifier of the feature set.
- `split`: Prediction split, defaulting to `latest`.
- `inputs`: Optional explicit input rows.
- `horizon`: Optional forecast horizon field in the request schema.

The service accepts `val`, `test`, and `latest` as split values. Actual feature and input validation also depends on the feature-set implementation.

### Configuration validation

Each model defines its default configuration and may define numeric parameter bounds. The shared validator merges defaults with supplied configuration values, rejects unknown parameters, and checks declared numeric bounds. Boolean values are not accepted as numeric hyperparameter values.

Model-specific validation may impose additional restrictions.

### Reproducibility and seed handling

The service defaults to seed `42` when a seed is not supplied. The selected seed is passed to the model implementation and returned in the training response where implemented.

Reproducibility depends on the model:

- Random Forest and Gradient Boosting pass the seed to scikit-learn.
- LSTM seeds PyTorch and NumPy and uses a seeded generator for training-row shuffling.
- QML seeds PyTorch and NumPy's random generator for initialization and subsampling.
- GARCH uses maximum-likelihood estimation and reports its fitting procedure as deterministic.
- Black–Scholes uses a trailing realised-volatility feature and does not use random training.

A seed alone does not guarantee identical results across every library version or execution environment.

### Resource limits and artifact integrity

Training runs in an isolated process with time and memory limits. The default timeout is 1,200 seconds and the default memory limit is 6,144 MB, configurable through `IFAP_JOB_MEM_MB`.

Model artifacts are hashed using SHA-256. Before a saved model is loaded for prediction, the service compares the artifact hash with the hash recorded in the registry. A mismatch raises `ARTIFACT_INTEGRITY_FAILED`.

Predictions containing non-finite values are rejected with `MODEL_FAILED`.

## 3. GARCH(1,1)

**Registered name:** `garch`

**Supported task:** `VOLATILITY`

### Purpose

GARCH(1,1) models time-varying financial return volatility. It estimates conditional variance from past innovations and variance, then produces a multi-step volatility forecast.

### Inputs and training procedure

- Uses the return series stored as `ret_all`.
- Restricts fitting to the training-period returns using `trainReturnEnd`.
- Multiplies returns by 100 before estimation.
- Fits a constant-mean GARCH(1,1) model with a normal innovation distribution.
- Uses maximum-likelihood estimation through the `arch` library.

### Parameters

The fitted parameters are:

- `mu`: Conditional mean.
- `omega`: Variance intercept.
- `alpha`: Response to the previous squared innovation.
- `beta`: Persistence from the previous conditional variance.
- `init_var`: Initial return variance.
- `nTrain`: Number of training returns.

The service uses a fixed GARCH order of `(1,1)` and does not expose additional model hyperparameters through its declared default configuration, which contains `distribution: normal`.

### Seed handling

The supplied seed is returned in training metadata, but the implementation does not use it in the maximum-likelihood fitting procedure.

### Forecasting and limitations

The prediction method recursively updates conditional variance and derives a horizon-based volatility estimate. It uses a persistence value capped at `0.9999` when computing the long-run variance.

The model assumes a normal innovation distribution and a constant conditional mean. Its forecasting quality depends on the return series and fitted parameters. The implementation is a baseline rather than a guarantee of accurate volatility forecasts.

## 4. Black–Scholes

**Registered name:** `black_scholes`

**Supported task:** `OPTION_PRICE`

### Purpose

Black–Scholes provides a baseline European option-pricing model using trailing realised volatility as the volatility estimate.

### Inputs

The prediction implementation reads:

- `S`: Underlying spot price.
- `log_moneyness`: Log-moneyness used to derive the strike.
- `tte`: Time to expiry.
- `rate`: Interest rate.
- The configured volatility feature.
- `is_call`: Determines call or put pricing using a threshold of `0.5`.

The strike is derived as `K = S * exp(-log_moneyness)`.

### Configuration

The default configuration is:

- `vol_feature: auto`

When `auto` is selected, the model chooses the realised-volatility feature associated with the longest configured trailing window. The selected feature must exist in the feature set; otherwise, the service raises `CONFIG_INVALID`.

### Seed handling

No random training procedure is used. The supplied seed does not affect the Black–Scholes calculation.

### Limitations

This is a baseline implementation using a trailing realised-volatility estimate. It prices European options and does not, by itself, model early exercise or guarantee accurate market prices.

## 5. Random Forest

**Registered name:** `random_forest`

**Supported tasks:** `VOLATILITY`, `OPTION_PRICE`

### Purpose

Random Forest is an ensemble regression model that combines predictions from multiple decision trees.

### Inputs and training procedure

- Uses the shared feature set's training split.
- Scales the training features through the feature-set scaling function.
- Scales the target using the model service's target-scale function.
- Fits a scikit-learn `RandomForestRegressor`.
- Saves the trained model as `model.joblib`.

### Hyperparameters

| Parameter | Default | Allowed range |
|---|---:|---:|
| `n_estimators` | 200 | 10–1000 |
| `max_depth` | 8 | 1–50 |
| `min_samples_leaf` | 3 | 1–100 |

### Seed handling

The supplied seed is passed as `random_state`. The implementation also sets `n_jobs=1`, limiting model fitting to a single job.

### Prediction and limitations

Predictions are transformed back using the target-scale function. Model behavior depends on the training data, selected features, and hyperparameters. Tree depth and minimum leaf size influence model complexity and overfitting.

## 6. Gradient Boosting

**Registered name:** `gradient_boosting`

**Supported tasks:** `VOLATILITY`, `OPTION_PRICE`

### Purpose

Gradient Boosting combines regression trees sequentially, with subsequent trees correcting errors made by the current ensemble.

### Inputs and training procedure

- Uses the training split of the shared feature set.
- Scales features and targets.
- Fits a scikit-learn `GradientBoostingRegressor`.
- Saves the trained model as `model.joblib`.

### Hyperparameters

| Parameter | Default | Allowed range |
|---|---:|---:|
| `n_estimators` | 250 | 10–1000 |
| `learning_rate` | 0.05 | 0.001–1.0 |
| `max_depth` | 3 | 1–12 |

### Seed handling

The supplied seed is passed as `random_state`.

### Prediction and limitations

Predictions are transformed back to the original target scale. Increasing the number or depth of trees can increase training cost and overfitting risk. Learning rate and estimator count should be considered together.

## 7. Long Short-Term Memory (LSTM)

**Registered name:** `lstm`

**Supported tasks:** `VOLATILITY`, `OPTION_PRICE`

### Purpose and architecture

LSTM is a recurrent neural network that learns patterns from sequences of recent daily features. For option-pricing tasks, the model combines the final LSTM hidden state with static contract features.

The network consists of:

1. An LSTM sequence layer.
2. A concatenation of the final hidden state and static features.
3. A dense layer with 16 units and ReLU activation.
4. A final linear output layer for regression.

### Hyperparameters

| Parameter | Default | Allowed range |
|---|---:|---:|
| `seq_len` | 20 | 2–120 |
| `hidden` | 32 | 4–256 |
| `epochs` | 40 | 1–500 |
| `batch_size` | 64 | 8–1024 |
| `lr` | 0.003 | 0.00001–0.5 |
| `patience` | 6 | 1–100 |
| `holdout_fraction` | 0.15 | 0.05–0.4 |

### Training and seed handling

The model uses only the training split. PyTorch and NumPy are seeded, deterministic PyTorch algorithms are enabled, and training-row shuffling uses a seeded generator.

The target is standardized during training. Adam optimization minimizes mean squared error.

Early stopping evaluates the chronologically last portion of the training rows. The best model weights are restored when training stops. The implementation uses a minimum improvement threshold of `1e-4` and reports the number of epochs run, best holdout loss, training duration, and execution device.

The implementation runs on CPU with PyTorch restricted to two threads.

### Prediction and limitations

Prediction reconstructs the saved network, generates sequence and static-feature tensors, and reverses target standardization and target scaling.

The model requires sufficient sequence data and compatible feature dimensions. Training time and overfitting depend on sequence length, hidden size, batch size, learning rate, and available training data.

## 8. Variational Quantum Regressor (QML)

**Registered name:** `qml_vqr`

**Supported tasks:** `VOLATILITY`, `OPTION_PRICE`

### Purpose and architecture

The QML model is a variational quantum regressor implemented using PennyLane and PyTorch. It runs on the `default.qubit` state-vector simulator on a classical CPU.

The circuit uses:

1. Angle embedding with Y-axis rotations.
2. `StronglyEntanglingLayers` with trainable weights.
3. An expectation-value measurement of Pauli Z on qubit 0.
4. A trainable affine mapping, `a * output + b`, to produce the regression output.

The implementation does not require quantum hardware and does not imply quantum advantage.

### Hyperparameters

| Parameter | Default | Allowed range |
|---|---:|---:|
| `n_qubits` | 4 | 1–64, subject to the simulator limit |
| `layers` | 3 | 1–6 |
| `epochs` | 120 | 1–300 |
| `lr` | 0.1 | 0.0001–1.0 |
| `max_train_rows` | 400 | 20–5000 |

### Inputs and training procedure

- Uses the training split of the shared feature set.
- Scales the features and selects the first `n_qubits` columns.
- Applies `2 * arctan(x)` to map input values into the interval `(-π, π)`.
- Standardizes the target.
- Trains circuit weights and affine output parameters using Adam and mean squared error.
- Saves model weights and normalization information in `model.pt`.

If the training data exceeds `max_train_rows`, the model selects a seeded, order-preserving subset of rows.

### Seed handling

The supplied seed initializes PyTorch and NumPy's random generator. The NumPy generator controls circuit-weight initialization and row subsampling.

### Limits and limitations

The default simulator limit is 8 qubits, configurable through `IFAP_MAX_QUBITS`. The requested qubit count must not exceed this limit or the number of available features.

The implementation uses a state-vector simulator, so computational cost can rise substantially as qubit count increases. `max_train_rows` helps bound simulation cost.

The model performs a fixed number of training epochs without the LSTM's early-stopping procedure. Its results depend on initialization, feature selection, circuit configuration, and training data. Simulator execution should not be interpreted as evidence of quantum speedup or quantum advantage.

## 9. Error Handling and Operational Constraints

The shared service validates model names, supported tasks, configurations, and prediction split values.

Relevant error conditions include:

- `CONFIG_INVALID`: Unknown model, unsupported task, invalid configuration, or an invalid volatility feature.
- `QUBIT_LIMIT_EXCEEDED`: Requested QML qubit count exceeds the configured simulator limit.
- `MODEL_NOT_FOUND`: The requested model artifact cannot be located.
- `ARTIFACT_INTEGRITY_FAILED`: The saved artifact does not match its registry hash.
- `VALIDATION_FAILED`: An unsupported prediction split is requested.
- `MODEL_FAILED`: Predictions contain non-finite values.

Training is isolated with configurable time and memory limits. The service verifies artifact integrity before loading models for prediction.

## 10. Reproducibility Checklist

For repeatable experiments:

1. Record the model name, feature-set identifier, configuration, and seed.
2. Preserve the training response and reported model metadata.
3. Record relevant dependency versions from the service description.
4. Keep the same feature set, preprocessing, and training/test split.
5. Account for possible differences across dependency versions and execution environments.
6. Never bypass artifact-integrity checks when loading a saved model.

## 11. Implementation References

- `common/ifap_common/modelservice.py`: Shared API contract, configuration validation, isolated training, resource limits, and artifact integrity.
- `services/classical/main.py`: Classical service model registration.
- `services/classical/models.py`: GARCH, Black–Scholes, Random Forest, and Gradient Boosting implementations.
- `services/dl/models.py`: LSTM implementation.
- `services/qml/models.py`: Variational quantum regressor implementation.
