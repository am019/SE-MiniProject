import torch  # noqa: F401  (warm import so the first prediction meets NFR-02)
from ifap_common.modelservice import create_model_app

app = create_model_app("dl", "dl", {"lstm": "services.dl.models:Lstm"}, extra_libs=[])
