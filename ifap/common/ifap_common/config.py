"""Configuration helpers: environment, secrets, data directory and service discovery.

SR-04: secrets are never hard-coded. They come from environment variables; if a
variable is absent a random value is generated once and persisted in the data
volume (never in the repository).
"""
from __future__ import annotations

import json
import os
import re
import secrets as _secrets
import time
from pathlib import Path


def env(name: str, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, default))
    except (TypeError, ValueError):
        return default


def data_dir() -> Path:
    p = Path(env("IFAP_DATA_DIR", "./var")).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def subdir(*parts: str) -> Path:
    p = data_dir().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- secrets
_GENERATORS = {
    "IFAP_JWT_SECRET": lambda: _secrets.token_urlsafe(48),
    "IFAP_SERVICE_TOKEN": lambda: _secrets.token_urlsafe(32),
    "IFAP_ADMIN_PASSWORD": lambda: _secrets.token_urlsafe(9),
    "IFAP_ANALYST_PASSWORD": lambda: _secrets.token_urlsafe(9),
}
_cache: dict[str, str] = {}


def _secrets_path() -> Path:
    return data_dir() / ".secrets.json"


def _load_or_create_secrets() -> dict:
    path = _secrets_path()
    for _ in range(100):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05)  # another process is still writing it
                continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({k: g() for k, g in _GENERATORS.items()}))
        return json.loads(path.read_text())
    raise RuntimeError("could not initialise secrets store")


def get_secret(name: str) -> str:
    if name in _cache:
        return _cache[name]
    v = env(name)
    if v is None:
        stored = _load_or_create_secrets()
        v = stored.get(name) or _GENERATORS[name]()
    _cache[name] = v
    return v


def service_token() -> str:
    return get_secret("IFAP_SERVICE_TOKEN")


# ----------------------------------------------------------- service discovery
BASE_SERVICES = ["ingestion", "features", "evaluation", "registry", "orchestrator", "report"]


def expand_env(s: str) -> str:
    """Expand ${VAR} and ${VAR:-default} in a string."""
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}",
                  lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), s)


def models_config_path() -> Path:
    return Path(env("IFAP_MODELS_CONFIG", "config/models.json"))


def load_models_config() -> dict:
    cfg = json.loads(models_config_path().read_text())
    cfg["services"] = {k: expand_env(v) for k, v in cfg.get("services", {}).items()}
    return cfg


def service_url(name: str) -> str:
    """URL of an internal service. Model services are resolved from models.json."""
    v = env(f"IFAP_URL_{name.upper()}")
    if v:
        return v.rstrip("/")
    if name not in BASE_SERVICES:
        try:
            u = load_models_config()["services"].get(name)
            if u:
                return u.rstrip("/")
        except (OSError, ValueError):
            pass
    return f"http://{name}:8000"


def all_service_urls() -> dict[str, str]:
    urls = {n: service_url(n) for n in BASE_SERVICES}
    try:
        for n, u in load_models_config()["services"].items():
            urls[n] = u.rstrip("/")
    except (OSError, ValueError):
        pass
    return urls
