"""FastAPI app factory shared by all internal services.

* NFR-08: structured JSON logs with a correlation ID on every request.
* SR-02: internal services require the X-Service-Token header (except /health).
* SR-06: unexpected exceptions return a generic INTERNAL_ERROR body.
"""
from __future__ import annotations

import hmac
import json
import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, ctx
from .config import service_token
from .errors import AppError, error_body

_logger = logging.getLogger("ifap")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
             "level": record.levelname, "service": getattr(record, "service", "-"),
             "requestId": ctx.request_id.get(), "msg": record.getMessage()}
        for k in ("method", "path", "status", "ms"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)  # server log only, never returned
        return json.dumps(d)


def setup_logging(service: str) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_JsonFormatter())
    h.addFilter(lambda r: setattr(r, "service", service) or True)
    _logger.handlers[:] = [h]
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def log(msg: str, level: int = logging.INFO, **kw) -> None:
    _logger.log(level, msg, extra=kw)


def create_app(name: str, internal: bool = True) -> FastAPI:
    setup_logging(name)
    app = FastAPI(title=f"IFAP {name}", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.name = name
    app.state.started = time.time()

    @app.middleware("http")
    async def _mw(request: Request, call_next):
        rid = request.headers.get("x-request-id") or ctx.new_request_id()
        token = ctx.request_id.set(rid)
        t0 = time.time()
        try:
            if internal and request.url.path != "/health":
                given = request.headers.get("x-service-token", "")
                if not hmac.compare_digest(given.encode(), service_token().encode()):
                    resp = JSONResponse(error_body("UNAUTHENTICATED", "Missing or invalid service token.", rid),
                                        status_code=401)
                    resp.headers["X-Request-ID"] = rid
                    log("rejected: bad service token", logging.WARNING, method=request.method,
                        path=request.url.path, status=401)
                    return resp
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            log("request", method=request.method, path=request.url.path,
                status=response.status_code, ms=int((time.time() - t0) * 1000))
            return response
        finally:
            ctx.request_id.reset(token)

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        rid = ctx.request_id.get()
        return JSONResponse(error_body(exc.code, exc.message, rid, exc.fields, exc.extra),
                            status_code=exc.status, headers={"X-Request-ID": rid})

    @app.exception_handler(RequestValidationError)
    async def _val_error(request: Request, exc: RequestValidationError):
        rid = ctx.request_id.get()
        fields = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "issue": e["msg"]}
                  for e in exc.errors()]
        return JSONResponse(error_body("VALIDATION_FAILED", "Request validation failed.", rid, fields),
                            status_code=422, headers={"X-Request-ID": rid})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        rid = ctx.request_id.get()
        code = {404: "NOT_FOUND", 405: "NOT_FOUND", 401: "UNAUTHENTICATED", 403: "FORBIDDEN"}.get(
            exc.status_code, "INTERNAL_ERROR")
        return JSONResponse(error_body(code, "Resource not available." if code == "NOT_FOUND"
                                       else "Request could not be processed.", rid),
                            status_code=exc.status_code, headers={"X-Request-ID": rid})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = ctx.request_id.get()
        _logger.error("unhandled exception", exc_info=exc)
        return JSONResponse(error_body("INTERNAL_ERROR", "An internal error occurred.", rid),
                            status_code=500, headers={"X-Request-ID": rid})

    @app.get("/health")
    def health():
        return {"status": "UP", "service": name, "version": __version__,
                "uptimeSeconds": int(time.time() - app.state.started)}

    return app
