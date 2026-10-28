"""Consistent error model (SRS section 3.7 of Part B). SR-06: errors never leak internals."""
from __future__ import annotations

from typing import Optional

DEFAULT_STATUS = {
    "UNAUTHENTICATED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "FILE_TYPE_UNSUPPORTED": 415,
    "FILE_TOO_LARGE": 413,
    "SCHEMA_INVALID": 422,
    "VALIDATION_FAILED": 422,
    "CONFIG_INVALID": 422,
    "QUBIT_LIMIT_EXCEEDED": 422,
    "DATASET_NOT_FOUND": 404,
    "EXPERIMENT_NOT_FOUND": 404,
    "MODEL_NOT_FOUND": 404,
    "NO_CHAMPION": 404,
    "NO_PREVIOUS_CHAMPION": 404,
    "EXPERIMENT_IN_PROGRESS": 409,
    "EXPERIMENT_NOT_COMPLETE": 409,
    "RATE_LIMITED": 429,
    "MODEL_FAILED": 500,
    "MODEL_TIMEOUT": 504,
    "ARTIFACT_INTEGRITY_FAILED": 500,
    "SERVICE_UNAVAILABLE": 503,
    "INTERNAL_ERROR": 500,
}


class AppError(Exception):
    def __init__(self, code: str, message: str, status: Optional[int] = None,
                 fields: Optional[list] = None, extra: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status or DEFAULT_STATUS.get(code, 400)
        self.fields = fields
        self.extra = extra or {}


def error_body(code: str, message: str, request_id: str, fields=None, extra=None) -> dict:
    body = {"code": code, "message": message, "requestId": request_id}
    if fields:
        body["fields"] = fields
    if extra:
        body.update(extra)
    return {"error": body}
