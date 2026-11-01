"""Stub model service (NFR-06 extensibility demonstration / TC-23)."""
from ifap_common.modelservice import create_model_app

app = create_model_app("stub", "stub", {"stub_persistence": "services.stub.models:Persistence"})
