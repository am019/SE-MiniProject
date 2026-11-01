"""Service-to-service HTTP client (adds service token + correlation ID, maps remote errors)."""
from __future__ import annotations

import httpx

from . import ctx
from .config import service_token, service_url
from .errors import AppError


def _headers() -> dict:
    return {"X-Service-Token": service_token(), "X-Request-ID": ctx.request_id.get()}


def call(service: str, method: str, path: str, json=None, params=None, timeout: float = 30.0,
         base_url: str | None = None, content: bytes | None = None):
    url = (base_url or service_url(service)).rstrip("/") + path
    try:
        r = httpx.request(method, url, json=json, content=content, params=params, headers=_headers(),
                          timeout=timeout)
    except httpx.TimeoutException:
        raise AppError("MODEL_TIMEOUT" if service not in ("registry", "ingestion", "features",
                                                           "evaluation", "report", "orchestrator")
                       else "SERVICE_UNAVAILABLE", f"Service '{service}' timed out.")
    except httpx.HTTPError:
        raise AppError("SERVICE_UNAVAILABLE", f"Service '{service}' is unavailable.")
    if r.status_code >= 400:
        try:
            e = r.json()["error"]
            raise AppError(e.get("code", "INTERNAL_ERROR"), e.get("message", "Upstream error."),
                           status=r.status_code, fields=e.get("fields"))
        except (ValueError, KeyError, TypeError):
            raise AppError("SERVICE_UNAVAILABLE", f"Service '{service}' returned an error.", status=502)
    if r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return r.content


def health(service: str, base_url: str | None = None, timeout: float = 2.0) -> dict:
    url = (base_url or service_url(service)).rstrip("/") + "/health"
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return {"status": "DOWN", "service": service}


def call_raw(service: str, path: str, params=None, timeout: float = 60.0):
    """GET returning (bytes, content_type) for file downloads; errors mapped like call()."""
    url = service_url(service).rstrip("/") + path
    try:
        r = httpx.get(url, params=params, headers=_headers(), timeout=timeout)
    except httpx.HTTPError:
        raise AppError("SERVICE_UNAVAILABLE", f"Service '{service}' is unavailable.")
    if r.status_code >= 400:
        try:
            e = r.json()["error"]
            raise AppError(e.get("code", "INTERNAL_ERROR"), e.get("message", "Upstream error."), status=r.status_code)
        except (ValueError, KeyError, TypeError):
            raise AppError("SERVICE_UNAVAILABLE", f"Service '{service}' returned an error.", status=502)
    return r.content, r.headers.get("content-type", "application/octet-stream")
