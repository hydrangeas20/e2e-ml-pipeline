"""Deterministic, opt-in fault injection.

Disabled unless ``MLPIPE_ENABLE_FAULT_INJECTION=true``. Faults are scoped to a model
version where that makes sense and count only live requests served by that version,
so a candidate passes pre-promotion validation and fails afterwards, by construction.

Kinds:
  error_after_n   requests served by ``model_version`` raise after the first ``n``
  feature_defect  apply ``features.DEFECTS[defect]`` before predicting
  audit_failure   prediction audit writes raise OSError
  slow            sleep ``seconds`` inside inference (overload / timeout tests)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from mlpipe.features import DEFECTS

FaultKind = Literal["error_after_n", "feature_defect", "audit_failure", "slow"]


class InjectedFault(RuntimeError):
    pass


@dataclass
class Fault:
    kind: FaultKind
    model_version: str | None = None
    n: int = 0
    defect: str | None = None
    seconds: float = 0.0
    hits: int = field(default=0)

    def public(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class FaultInjector:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._faults: dict[str, Fault] = {}
        self._lock = threading.Lock()

    def set(self, fault: Fault) -> None:
        if not self.enabled:
            raise PermissionError("fault injection is disabled")
        if fault.kind == "feature_defect" and fault.defect not in DEFECTS:
            raise ValueError(f"unknown defect {fault.defect!r}; choose from {sorted(DEFECTS)}")
        if fault.kind == "slow" and not 0 < fault.seconds <= 30:
            raise ValueError("slow.seconds must be in (0, 30]")
        with self._lock:
            self._faults[fault.kind] = fault

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [f.public() for f in self._faults.values()]

    def _get(self, kind: str, version: str | None) -> Fault | None:
        f = self._faults.get(kind)
        if f is None or (f.model_version is not None and f.model_version != version):
            return None
        return f

    def feature_defect(self, version: str) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            f = self._get("feature_defect", version)
            return f.defect if f else None

    def before_predict(self, version: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            slow = self._get("slow", version)
            err = self._get("error_after_n", version)
            raise_now = False
            if err is not None:
                err.hits += 1
                raise_now = err.hits > err.n
        if slow is not None:
            time.sleep(slow.seconds)
        if raise_now:
            raise InjectedFault(f"injected error_after_n (n={err.n}) for {version}")  # type: ignore[union-attr]

    def audit_should_fail(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return "audit_failure" in self._faults
