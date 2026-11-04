"""Child process for crash-consistency tests: ``python -m tests.crash_harness REG ACTION``.

The parent sets ``MLPIPE_CRASH_AT``; the process dies with ``os._exit(86)`` at that
point (see mlpipe/crash.py). Exit 0 means the action completed without crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mlpipe.config import Settings
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.deployment import DeploymentController
from mlpipe.serving.observability import Metrics


def main(reg_dir: str, action: str, arg: str = "") -> int:
    reg = Registry(Path(reg_dir))
    settings = Settings(
        registry_dir=Path(reg_dir),
        gate_min_samples=5,
        gate_window=50,
        gate_max_consecutive_failures=3,
    )
    ctl = DeploymentController(reg, SERVING_SCHEMA, settings, Metrics())
    ctl.startup()
    if action == "promote":
        ctl.promote(arg or "v2-rf")
    elif action == "promote_then_trip":  # promote, then fail requests until rollback
        ctl.promote(arg or "v2-rf")
        lm = ctl.current()
        assert lm is not None
        for _ in range(10):
            ctl.record_outcome(lm, False, 1.0)
    elif action == "register":
        reg.register(Path(arg))
    else:
        raise SystemExit(f"unknown action {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
