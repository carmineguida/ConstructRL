"""Diagnostic logging utilities for Construct RL."""

import time

from .config import config

_T0 = time.perf_counter()
_VERBOSE = config["verbose"]
_SLOW_MS = config["slow_ms"]
_step_counter = 0


def _ts():
    return f"{time.perf_counter()-_T0:10.3f}s"


def dlog(tag, msg, force=False):
    if _VERBOSE or force:
        print(f"[{_ts()}][{tag}] {msg}", flush=True)


def dlog_step(tag, msg):
    """Per-step log, printed every step (set _VERBOSE=False to mute)."""
    global _step_counter
    _step_counter += 1
    if _VERBOSE:
        print(f"[{_ts()}][{tag}] {msg}", flush=True)
