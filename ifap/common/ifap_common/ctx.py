"""Request-scoped correlation ID (NFR-08)."""
import contextvars
import uuid

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    return "req-" + uuid.uuid4().hex[:10]
