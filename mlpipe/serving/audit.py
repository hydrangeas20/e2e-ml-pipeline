"""Prediction audit log (JSONL, one record per returned prediction).

When ``required`` is true a prediction is returned only after its record has been
written (and fsync'd if ``fsync``); a write failure turns the response into a 503.
Records contain identifiers and outputs only, never feature values.

Durability: without ``fsync`` the record is in the OS page cache, not on disk, when
the response is sent. Single writer per file; not safe for multi-process serving.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditError(Exception):
    pass


class PredictionAuditLog:
    def __init__(self, path: Path, *, fsync: bool, fail_hook: Callable[[], bool]) -> None:
        self.path = path
        self.fsync = fsync
        self._fail_hook = fail_hook
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: list[dict[str, Any]]) -> None:
        ts = datetime.now(UTC).isoformat()
        data = "".join(json.dumps({"ts": ts, **r}, sort_keys=True) + "\n" for r in records)
        with self._lock:
            try:
                if self._fail_hook():
                    raise OSError("injected audit failure")
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
                try:
                    offset = os.lseek(fd, 0, os.SEEK_END)
                    try:
                        n = os.write(fd, data.encode())
                        if n != len(data.encode()):
                            raise OSError(f"short write {n}")
                        if self.fsync:
                            os.fsync(fd)
                    except OSError:
                        os.ftruncate(fd, offset)  # all-or-nothing
                        raise
                finally:
                    os.close(fd)
            except OSError as e:
                raise AuditError(str(e)) from e
