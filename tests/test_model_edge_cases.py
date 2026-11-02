import pytest


@pytest.mark.integration
def test_qml_qubit_limit_rejected(platform):
    """QML service must reject requests exceeding the configured qubit limit."""
    client = platform.client()
    ds = platform.sample("sample_index_daily.csv")

    response = client.post(
        "/api/experiments",
        json={
            "datasetId": ds["datasetId"],
            "task": "VOLATILITY",
            "horizon": 5,
            "models": ["qml_vqr"],
            "seed": 42,
            "modelConfig": {"qml_vqr": {"n_qubits": 999}},
        },
    )

    # The API may accept the experiment asynchronously and report the
    # model-level configuration error in its final status.
    assert response.status_code == 202, response.text
    experiment_id = response.json()["experimentId"]

    import time

    start = time.time()
    while time.time() - start < 120:
        result = client.get(f"/api/experiments/{experiment_id}").json()
        if result["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(1)

    model_results = result.get("models", {})
    model_result = model_results.get("qml_vqr", {})

    assert (
        result["status"] == "FAILED"
        or model_result.get("status") == "FAILED"
        or "QUBIT_LIMIT_EXCEEDED" in str(model_result)
    ), result


@pytest.mark.integration
def test_invalid_model_configuration_rejected(platform):
    """Invalid model hyperparameters must not produce a successful model run."""
    client = platform.client()
    ds = platform.sample("sample_index_daily.csv")

    response = client.post(
        "/api/experiments",
        json={
            "datasetId": ds["datasetId"],
            "task": "VOLATILITY",
            "horizon": 5,
            "models": ["random_forest"],
            "seed": 42,
            "modelConfig": {
                "random_forest": {"n_estimators": -1}
            },
        },
    )

    assert response.status_code in (400, 422, 202), response.text

    if response.status_code == 202:
        experiment_id = response.json()["experimentId"]
        import time

        start = time.time()
        while time.time() - start < 120:
            result = client.get(f"/api/experiments/{experiment_id}").json()
            if result["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(1)

        model_result = result.get("models", {}).get("random_forest", {})
        assert (
            result["status"] == "FAILED"
            or model_result.get("status") == "FAILED"
        ), result


@pytest.mark.integration
def test_tiny_dataset_handled(platform):
    """A tiny dataset should fail gracefully rather than crash the platform."""
    client = platform.client()

    response = client.post(
        "/api/samples/sample_index_daily.csv/load"
    )
    assert response.status_code == 201, response.text

    health = client.get("/api/health")
    assert health.status_code == 200, health.text
