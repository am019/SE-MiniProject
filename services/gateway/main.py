"""API Gateway (SR-01, SR-02, SR-07, SR-08, SR-03, SR-06) + static dashboard host.

The only component published outside the private network. Everything below /api needs a JWT
(except /api/auth/login); internal services are reached with the service token.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import jwt
from fastapi import Body, Depends, File, Request, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ifap_common import client
from ifap_common.appfactory import create_app
from ifap_common.config import all_service_urls, env_int, get_secret
from ifap_common.errors import AppError

app = create_app("gateway", internal=False)
MAX_BYTES = env_int("IFAP_MAX_UPLOAD_MB", 25) * 1024 * 1024
RATE_LIMIT = env_int("IFAP_RATE_LIMIT", 60)
TTL_MIN = env_int("IFAP_JWT_TTL_MIN", 60)
ALGO = "HS256"


# ------------------------------------------------------------------ security headers
@app.middleware("http")
async def _sec_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Content-Security-Policy",
                            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                            "connect-src 'self'; frame-ancestors 'none'")
    if request.url.path.startswith("/api"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


# ------------------------------------------------------------------ rate limiting (SR-07)
class RateLimiter:
    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.hits: dict = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                retry = max(1, int(self.window - (now - q[0])) + 1)
                raise AppError("RATE_LIMITED", "Too many requests; please slow down.", extra={"retryAfterSec": retry})
            q.append(now)


user_limiter = RateLimiter(RATE_LIMIT)
login_limiter = RateLimiter(max(10, RATE_LIMIT // 3))


@app.exception_handler(AppError)
async def _app_error(request: Request, exc: AppError):
    from fastapi.responses import JSONResponse
    from ifap_common import ctx
    from ifap_common.errors import error_body
    rid = ctx.request_id.get()
    headers = {"X-Request-ID": rid}
    if exc.code == "RATE_LIMITED":
        headers["Retry-After"] = str(exc.extra.get("retryAfterSec", 60))
    return JSONResponse(error_body(exc.code, exc.message, rid, exc.fields,
                                   {k: v for k, v in exc.extra.items() if k != "retryAfterSec"}),
                        status_code=exc.status, headers=headers)


# ------------------------------------------------------------------ auth (SR-01, SR-08)
class User(BaseModel):
    name: str
    role: str


def _users() -> dict:
    return {"admin": ("Administrator", get_secret("IFAP_ADMIN_PASSWORD")),
            "analyst": ("Analyst", get_secret("IFAP_ANALYST_PASSWORD"))}


class LoginReq(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(req: LoginReq, request: Request):
    login_limiter.check("ip:" + (request.client.host if request.client else "?"))
    rec = _users().get(req.username)
    ok = rec is not None and hmac.compare_digest(req.password.encode(), rec[1].encode())
    if not ok:
        raise AppError("UNAUTHENTICATED", "Invalid username or password.")
    now = int(time.time())
    token = jwt.encode({"sub": req.username, "role": rec[0], "iat": now, "exp": now + TTL_MIN * 60},
                       get_secret("IFAP_JWT_SECRET"), algorithm=ALGO)
    return {"token": token, "tokenType": "Bearer", "expiresInSec": TTL_MIN * 60, "role": rec[0],
            "username": req.username}


def auth(request: Request) -> User:
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer ") or len(h) < 15:
        raise AppError("UNAUTHENTICATED", "Authentication required.")
    try:
        claims = jwt.decode(h[7:].strip(), get_secret("IFAP_JWT_SECRET"), algorithms=[ALGO],
                            options={"require": ["exp", "sub", "role"]})
    except jwt.PyJWTError:
        raise AppError("UNAUTHENTICATED", "Invalid or expired token.")
    if claims.get("role") not in ("Analyst", "Administrator"):
        raise AppError("UNAUTHENTICATED", "Invalid or expired token.")
    user_limiter.check("user:" + str(claims["sub"]))
    return User(name=claims["sub"], role=claims["role"])


def admin(u: User = Depends(auth)) -> User:
    if u.role != "Administrator":
        raise AppError("FORBIDDEN", "This action requires the Administrator role.")
    return u


@app.get("/api/me")
def me(u: User = Depends(auth)):
    return u


# ------------------------------------------------------------------ datasets (SR-03)
@app.post("/api/datasets", status_code=201)
async def upload(request: Request, file: UploadFile = File(...), u: User = Depends(auth)):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BYTES + 1_000_000:
        raise AppError("FILE_TOO_LARGE", f"File exceeds the {MAX_BYTES // (1024 * 1024)} MB limit.")
    name = file.filename or ""
    if not name.lower().endswith(".csv"):
        raise AppError("FILE_TYPE_UNSUPPORTED", "Only .csv files are accepted.")  # before any parsing
    buf = bytearray()
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_BYTES:
            raise AppError("FILE_TOO_LARGE", f"File exceeds the {MAX_BYTES // (1024 * 1024)} MB limit.")
    return client.call("ingestion", "POST", "/datasets", params={"filename": name}, content=bytes(buf), timeout=120)


_SAMPLES = {"sample_index_daily.csv": "Synthetic daily index prices (volatility task)",
            "sample_option_chain.csv": "Synthetic option chain (option-price task)"}
_sample_dir = Path(__file__).resolve().parents[2] / "data" / "sample"


@app.get("/api/samples")
def samples(u: User = Depends(auth)):
    return {"samples": [{"file": f, "description": d, "label": "SYNTHETIC SAMPLE - not real market data"}
                        for f, d in _SAMPLES.items() if (_sample_dir / f).exists()]}


@app.post("/api/samples/{name}/load", status_code=201)
def load_sample(name: str, u: User = Depends(auth)):
    if name not in _SAMPLES or not (_sample_dir / name).exists():  # whitelist: no path traversal
        raise AppError("NOT_FOUND", "Sample not available.")
    return client.call("ingestion", "POST", "/datasets", params={"filename": name},
                       content=(_sample_dir / name).read_bytes(), timeout=120)


@app.get("/api/datasets")
def datasets(u: User = Depends(auth)):
    return client.call("ingestion", "GET", "/datasets")


@app.get("/api/datasets/{ds_id}")
def dataset(ds_id: str, u: User = Depends(auth)):
    return client.call("ingestion", "GET", f"/datasets/{ds_id}")


@app.get("/api/datasets/{ds_id}/summary")
def dataset_summary(ds_id: str, u: User = Depends(auth)):
    return client.call("ingestion", "GET", f"/datasets/{ds_id}/summary")


# ------------------------------------------------------------------ experiments
@app.get("/api/models/available")
def models_available(u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/models/available", timeout=10)


@app.post("/api/experiments", status_code=202)
def start_experiment(body: dict = Body(...), u: User = Depends(auth)):
    return client.call("orchestrator", "POST", "/experiments", json=body, timeout=60)


@app.get("/api/experiments")
def experiments(u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/experiments")


@app.get("/api/experiments/{eid}")
def experiment(eid: str, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", f"/experiments/{eid}")


@app.get("/api/experiments/{eid}/leaderboard")
def leaderboard(eid: str, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", f"/experiments/{eid}/leaderboard")


@app.get("/api/experiments/{eid}/selection")
def selection(eid: str, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", f"/experiments/{eid}/selection")


@app.post("/api/predictions")
def predictions(body: dict = Body(...), u: User = Depends(auth)):
    return client.call("orchestrator", "POST", "/predictions", json=body, timeout=30)


# ------------------------------------------------------------------ models / champion
@app.get("/api/models")
def models(experimentId: Optional[str] = None, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/models", params={"experimentId": experimentId} if experimentId else None)


@app.get("/api/models/champion")
def champion(task: str, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/champion", params={"task": task})


@app.get("/api/models/champions")
def champions(u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/champions")


@app.get("/api/models/champion/history")
def champion_history(task: Optional[str] = None, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", "/champion/history", params={"task": task} if task else None)


@app.post("/api/models/champion/rollback")
def rollback(body: dict = Body(...), u: User = Depends(admin)):
    return client.call("orchestrator", "POST", "/champion/rollback", json={"task": body.get("task")})


@app.get("/api/models/{mid}")
def model(mid: str, u: User = Depends(auth)):
    return client.call("orchestrator", "GET", f"/models/{mid}")


@app.delete("/api/models/{mid}")
def delete_model(mid: str, u: User = Depends(admin)):
    return client.call("orchestrator", "DELETE", f"/models/{mid}")


# ------------------------------------------------------------------ reports
@app.get("/api/reports/{eid}")
def report(eid: str, format: str = "json", u: User = Depends(auth)):
    body, mt = client.call_raw("report", f"/reports/{eid}", params={"format": format}, timeout=120)
    return Response(body, media_type=mt, headers={"Content-Disposition": f'attachment; filename="ifap-{eid}.{format}"'})


# ------------------------------------------------------------------ admin (SR-08)
def _probe(item):
    name, url = item
    t0 = time.perf_counter()
    h = client.health(name, base_url=url, timeout=1.0)
    return {"service": name, "status": "UP" if h.get("status") == "UP" else "DOWN",
            "latencyMs": round((time.perf_counter() - t0) * 1000, 1), "version": h.get("version"),
            "lastChecked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@app.get("/api/admin/health")
def admin_health(u: User = Depends(admin)):
    items = [(n, url) for n, url in all_service_urls().items()]
    with ThreadPoolExecutor(max_workers=min(16, len(items))) as ex:
        rows = list(ex.map(_probe, items))
    gw = {"service": "gateway", "status": "UP", "latencyMs": 0.0, "version": None,
          "lastChecked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return {"services": [gw] + sorted(rows, key=lambda r: r["service"]),
            "overall": "UP" if all(r["status"] == "UP" for r in rows) else "DEGRADED"}


@app.get("/api/admin/config")
def get_config(u: User = Depends(admin)):
    return client.call("orchestrator", "GET", "/config")


@app.put("/api/admin/config")
def put_config(body: dict = Body(...), u: User = Depends(admin)):
    return client.call("orchestrator", "PUT", "/config", json=body)


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
def _unknown(rest: str, u: User = Depends(auth)):
    raise AppError("NOT_FOUND", "Resource not available.")


# ------------------------------------------------------------------ dashboard (static)
_web = Path(__file__).resolve().parents[2] / "web"
if _web.exists():
    app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")
