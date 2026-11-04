"""Crash consistency: the deployment process is killed (os._exit, no cleanup) at named
transaction boundaries, then a *fresh process* restarts from durable state.

Covers process crashes only (data handed to the kernel survives). Power loss and
storage that ignores fsync are not simulated.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from mlpipe.config import Settings
from mlpipe.crash import CRASH_EXIT_CODE
from mlpipe.registry import Registry, State
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.deployment import DeploymentController, DeploymentError
from mlpipe.serving.observability import Metrics
from tests.conftest import ROOT


def _run(reg: Registry, action: str, crash_at: str | None, arg: str = "") -> int:
    env = {**os.environ, "MLPIPE_ENABLE_FAULT_INJECTION": "true"}
    env.pop("MLPIPE_CRASH_AT", None)
    if crash_at:
        env["MLPIPE_CRASH_AT"] = crash_at
    p = subprocess.run(
        [sys.executable, "-m", "tests.crash_harness", str(reg.root), action, arg],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode in (0, CRASH_EXIT_CODE), p.stdout[-2000:] + p.stderr[-2000:]
    return p.returncode


def _restart(reg: Registry, settings: Settings) -> DeploymentController:
    c = DeploymentController(reg, SERVING_SCHEMA, settings, Metrics())
    c.startup()
    return c


def _actives(reg: Registry) -> list[str]:
    return sorted(v for v, e in reg.read_state()["versions"].items() if e["state"] == "active")


def _serving(c: DeploymentController) -> str | None:
    lm = c.current()
    return lm.version if lm else None


# --- promotion boundaries ---------------------------------------------------------------


def test_crash_before_promote_commit(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    assert _run(bootstrapped, "promote", "promote:after_event") == CRASH_EXIT_CODE
    unc = bootstrapped.inspect()["uncommitted_events"]
    assert [e["action"] for e in unc] == ["promote"]  # journaled, never committed
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) == "v1-legacy-rf" and c.readiness()["ready"]
    # the stuck candidate is released, so the promotion can simply be retried
    assert bootstrapped.state_of("v2-rf") == State.VALIDATED
    assert c.promote("v2-rf")["result"] == "active_on_probation"
    gens = [e["generation"] for e in bootstrapped.events()]
    assert len(gens) == len(set(gens))
    assert [e["action"] for e in bootstrapped.inspect()["uncommitted_events"]] == ["promote"]


@pytest.mark.parametrize(
    "point",
    ["promote:after_commit", "promote:after_commit_before_activate", "promote:after_activate"],
)
def test_crash_after_promote_commit_restores_known_good(
    bootstrapped: Registry, settings_factory: Callable[..., Settings], point: str
) -> None:
    assert _run(bootstrapped, "promote", point) == CRASH_EXIT_CODE
    st = bootstrapped.read_state()
    assert st["active"] == "v2-rf" and st["probation"]["version"] == "v2-rf"
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) == "v1-legacy-rf" and c.readiness()["ready"]
    assert _actives(bootstrapped) == ["v1-legacy-rf"]
    last = bootstrapped.events()[-1]
    assert last["action"] == "rollback" and last["rollback_trigger"] == "restart_during_probation"
    with pytest.raises(DeploymentError, match="invalid_state"):
        c.promote("v2-rf")  # rolled back: cannot silently return


def test_clean_exit_during_probation_then_restart(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    assert _run(bootstrapped, "promote", None) == 0  # process ends mid-probation
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) == "v1-legacy-rf"
    assert bootstrapped.state_of("v2-rf") == State.ROLLED_BACK


# --- rollback boundaries ----------------------------------------------------------------


@pytest.mark.parametrize(
    "point,journaled",
    [
        ("rollback:after_swap", False),  # memory restored, nothing durable yet
        ("rollback:after_event", True),  # journaled, state.json not committed
    ],
)
def test_crash_during_automatic_rollback(
    bootstrapped: Registry,
    settings_factory: Callable[..., Settings],
    point: str,
    journaled: bool,
) -> None:
    assert _run(bootstrapped, "promote_then_trip", point) == CRASH_EXIT_CODE
    st = bootstrapped.read_state()
    assert st["active"] == "v2-rf" and st["probation"]  # durable state never moved
    unc = [e["action"] for e in bootstrapped.inspect()["uncommitted_events"]]
    assert unc == (["rollback"] if journaled else [])
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) == "v1-legacy-rf" and c.readiness()["ready"]
    assert _actives(bootstrapped) == ["v1-legacy-rf"]
    assert bootstrapped.state_of("v2-rf") == State.ROLLED_BACK
    rb = [e for e in bootstrapped.events() if e["action"] == "rollback"]
    committed = [e for e in rb if e not in bootstrapped.inspect()["uncommitted_events"]]
    assert [e["rollback_trigger"] for e in committed] == ["restart_during_probation"]


def test_crash_after_rollback_commit(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    assert _run(bootstrapped, "promote_then_trip", "rollback:after_commit") == CRASH_EXIT_CODE
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) == "v1-legacy-rf" and not c.on_probation
    rb = [e for e in bootstrapped.events() if e["action"] == "rollback"]
    assert len(rb) == 1 and rb[0]["rollback_trigger"] == "health_gate"


# --- artifacts on restart -----------------------------------------------------------------


def test_corrupt_active_artifact_on_restart_fails_closed(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    p = bootstrapped.artifact_dir("v1-legacy-rf") / "model.skops"
    b = bytearray(p.read_bytes())
    b[100] ^= 1
    p.write_bytes(bytes(b))
    c = _restart(bootstrapped, settings_factory())
    r = c.readiness()
    assert _serving(c) is None and not r["ready"] and "integrity" in str(r["reason"])


def test_missing_known_good_on_restart_during_probation_fails_closed(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    assert _run(bootstrapped, "promote", None) == 0
    shutil.rmtree(bootstrapped.artifact_dir("v1-legacy-rf"))
    c = _restart(bootstrapped, settings_factory())
    assert _serving(c) is None and "known-good" in str(c.readiness()["reason"])
    assert bootstrapped.read_state()["probation"]  # left intact for the operator


def test_crash_during_register_leaves_retryable_orphan(
    registry: Registry, artifacts: dict[str, Path], tmp_path: Path
) -> None:
    src = tmp_path / "v7"
    shutil.copytree(artifacts["v3-rf"], src)
    m = json.loads((src / "manifest.json").read_text())
    m["model_version"] = "v7"
    (src / "manifest.json").write_text(json.dumps(m))
    assert _run(registry, "register", "register:after_copy", str(src)) == CRASH_EXIT_CODE
    assert (registry.models_dir / "v7").exists() and registry.state_of("v7") is None
    assert registry.register(src) == "v7"


# --- real server --------------------------------------------------------------------------


@pytest.mark.slow
def test_server_killed_mid_rollback_recovers_on_restart(
    bootstrapped: Registry, tmp_path: Path
) -> None:
    token = "crash-admin-token-0123456789"
    env = {
        **os.environ,
        "MLPIPE_REGISTRY_DIR": str(bootstrapped.root),
        "MLPIPE_AUDIT_LOG_PATH": str(tmp_path / "p.jsonl"),
        "MLPIPE_ADMIN_TOKEN": token,
        "MLPIPE_ENABLE_FAULT_INJECTION": "true",
        "MLPIPE_GATE_MIN_SAMPLES": "5",
        "MLPIPE_GATE_MAX_CONSECUTIVE_FAILURES": "3",
    }

    def start(extra: dict[str, str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8797"],
            cwd=ROOT,
            env={**env, **extra},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def ready(c: httpx.Client) -> dict[str, Any]:
        for _ in range(100):
            try:
                r = c.get("/readyz")
                if r.status_code == 200:
                    return r.json()  # type: ignore[no-any-return]
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        raise AssertionError("not ready")

    from mlpipe.validation import smoke_frame

    rows = smoke_frame().head(1).to_dict(orient="records")
    c = httpx.Client(
        base_url="http://127.0.0.1:8797", timeout=10, headers={"Authorization": f"Bearer {token}"}
    )
    proc = start({"MLPIPE_CRASH_AT": "rollback:after_swap"})
    try:
        ready(c)
        assert c.post("/admin/promote", json={"version": "v2-rf"}).status_code == 200
        c.post("/admin/faults", json={"kind": "error_after_n", "model_version": "v2-rf", "n": 0})
        for _ in range(5):
            try:
                c.post("/v1/predict", json={"instances": rows})
            except httpx.TransportError:
                break
        assert proc.wait(timeout=20) == CRASH_EXIT_CODE
    finally:
        if proc.poll() is None:
            proc.kill()
    proc = start({})
    try:
        r = ready(c)
        assert r["model_version"] == "v1-legacy-rf"
        b = c.post("/v1/predict", json={"instances": rows}).json()
        assert b["model_version"] == "v1-legacy-rf"
        assert bootstrapped.state_of("v2-rf") == State.ROLLED_BACK
        audit = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text().splitlines()]
        assert all(a["model_version"] == "v1-legacy-rf" for a in audit), "no failed v2 audited"
    finally:
        c.close()
        proc.terminate()
        proc.wait(timeout=20)
