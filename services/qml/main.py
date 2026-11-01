import pennylane  # noqa: F401  (warm import so the first prediction meets NFR-02)
import torch  # noqa: F401
from ifap_common.modelservice import create_model_app

app = create_model_app("qml", "qml", {"qml_vqr": "services.qml.models:VariationalRegressor"},
                       extra_libs=["pennylane_lightning"])
