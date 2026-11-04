"""Deterministic process-crash points for crash-consistency tests.

``crashpoint(name)`` calls ``os._exit(CRASH_EXIT_CODE)`` (no cleanup, no finally
blocks, no atexit) when ``MLPIPE_CRASH_AT == name`` *and* fault injection is enabled
via ``MLPIPE_ENABLE_FAULT_INJECTION=true``. Otherwise it is a no-op.

This simulates a process crash. Data already handed to the kernel (written, or
fsync'd) survives, exactly as with SIGKILL. It does NOT simulate power loss.
"""

from __future__ import annotations

import os

CRASH_EXIT_CODE = 86


def crashpoint(name: str) -> None:
    if (
        os.environ.get("MLPIPE_CRASH_AT") == name
        and os.environ.get("MLPIPE_ENABLE_FAULT_INJECTION", "").lower() == "true"
    ):
        os._exit(CRASH_EXIT_CODE)
