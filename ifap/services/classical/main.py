"""Classical ML service (FR-06, FR-09). Implements the common model-service contract."""
from ifap_common.modelservice import create_model_app

app = create_model_app("classical", "classical", {
    "garch": "services.classical.models:Garch",
    "black_scholes": "services.classical.models:BlackScholes",
    "random_forest": "services.classical.models:RandomForest",
    "gradient_boosting": "services.classical.models:GradientBoosting",
}, extra_libs=["joblib"])
