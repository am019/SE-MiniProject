"""Isolated, time-limited execution of model training (FR-09, NFR-04, SR-08).

Training runs in a *separate spawned process* so a crash, hang or memory blow-up
cannot take down the model service. The parent enforces a wall-clock timeout and
the child applies an address-space limit where the OS supports it.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Callable

from .errors import AppError


def _child(q, fn, args, mem_mb):
    try:
        try:
            import resource
            if mem_mb:
                lim = int(mem_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:  # not available on this platform: timeout still applies
            pass
        q.put(("ok", fn(*args)))
    except AppError as e:
        q.put(("apperr", (e.code, e.message)))
    except MemoryError:
        q.put(("err", "memory limit exceeded"))
    except BaseException as e:  # noqa: BLE001 - report anything to the parent
        q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


def run_isolated(fn: Callable, args: tuple, timeout_s: float, mem_mb: int | None = None):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, fn, args, mem_mb), daemon=True)
    p.start()
    try:
        status, payload = q.get(timeout=timeout_s)
    except Exception:
        p.kill()
        p.join(5)
        if p.exitcode is not None and p.exitcode != 0 and p.exitcode != -9:
            raise AppError("MODEL_FAILED", "Model training process terminated unexpectedly.")
        raise AppError("MODEL_TIMEOUT", f"Model training exceeded the {int(timeout_s)} s time limit.")
    p.join(10)
    if p.is_alive():
        p.kill()
    if status == "ok":
        return payload
    if status == "apperr":
        raise AppError(payload[0], payload[1])
    # details stay in the restricted server log (SR-06); the caller only gets a safe message
    import logging
    logging.getLogger("ifap").error("isolated job failed: %s", str(payload)[:800])
    raise AppError("MODEL_FAILED", "Model training failed.")
