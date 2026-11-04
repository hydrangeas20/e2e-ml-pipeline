"""Regression tests for defects confirmed in the PR #1 reliability review.

Each test failed on a0c9a9c (see docs/REVIEW.md for the before/after record).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipe import registry as registry_mod
from mlpipe.config import Settings
from mlpipe.registry import Registry, RegistryError, State
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.app import create_app
from mlpipe.serving.deployment import DeploymentController, DeploymentError
from mlpipe.serving.observability import Metrics
from mlpipe.validation import validate_artifact
from tests.conftest import AUTH


def _ctl(reg: Registry, settings: Settings) -> DeploymentController:
    c = DeploymentController(reg, SERVING_SCHEMA, settings, Metrics())
    c.startup()
    return c


def _corrupt(p: Path) -> None:
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))


def _active_versions(reg: Registry) -> list[str]:
    return sorted(v for v, e in reg.read_state()["versions"].items() if e["state"] == "active")


# R1 -----------------------------------------------------------------------------------
def test_r1_restart_in_probation_with_unloadable_known_good_fails_closed(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    """An unproven candidate must not serve, without a gate, as its own known-good."""
    _ctl(bootstrapped, settings_factory()).promote("v2-rf")
    _corrupt(bootstrapped.artifact_dir("v1-legacy-rf") / "model.skops")
    fresh = _ctl(bootstrapped, settings_factory())
    assert fresh.current() is None
    r = fresh.readiness()
    assert not r["ready"] and "known-good" in str(r["reason"])
    assert bootstrapped.read_state()["probation"]["version"] == "v2-rf"  # left for operator


# R2 -----------------------------------------------------------------------------------
def test_r2_unpersisted_rollback_is_reconciled_before_next_promotion(
    bootstrapped: Registry,
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctl = _ctl(bootstrapped, settings_factory())
    ctl.promote("v2-rf")
    real = bootstrapped.apply_many

    def failing(*a: Any, **kw: Any) -> Any:
        if kw.get("action") == "rollback":
            raise OSError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(bootstrapped, "apply_many", failing)
    assert ctl.rollback(reason="op")["result"] == "rolled_back"
    cur = ctl.current()
    assert cur is not None and cur.version == "v1-legacy-rf"
    assert ctl.readiness()["degraded"]
    # still failing: promotion must be refused rather than built on stale durable state
    with pytest.raises(DeploymentError) as ei:
        ctl.promote("v3-rf")
    assert ei.value.code == "state_degraded"
    monkeypatch.undo()
    assert ctl.promote("v3-rf")["result"] == "active_on_probation"
    st = bootstrapped.read_state()
    assert _active_versions(bootstrapped) == ["v3-rf"]
    assert st["versions"]["v2-rf"]["state"] == "rolled_back"
    assert st["known_good"] == "v1-legacy-rf"


# R3 -----------------------------------------------------------------------------------
def test_r3_journal_generation_never_reused_after_failed_commit(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real = registry_mod.write_json_atomic

    def fail_once(*a: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("crash before commit")
        real(*a)

    monkeypatch.setattr(registry_mod, "write_json_atomic", fail_once)
    with pytest.raises(OSError):
        registry.transition("v2-rf", State.VALIDATED, action="validate")
    registry.transition("v3-rf", State.VALIDATED, action="validate")
    gens = [e["generation"] for e in registry.events() if "generation" in e]
    assert len(gens) == len(set(gens)), f"generation reused: {gens}"
    unc = registry.inspect()["uncommitted_events"]
    assert [e["model_version"] for e in unc] == ["v2-rf"]
    assert registry.state_of("v2-rf") == State.REGISTERED


# R4 -----------------------------------------------------------------------------------
def test_r4_timed_out_request_leaves_no_prediction_audit_record(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    c = client_factory(max_concurrency=1, max_queue=0, request_timeout_s=0.1)
    c.post("/admin/faults", json={"kind": "slow", "seconds": 0.4})
    r = c.post("/v1/predict", json={"instances": rows[:1]}, headers={"X-Request-ID": "late-1"})
    assert r.status_code == 504
    time.sleep(0.6)  # abandoned computation finishes in the background
    audit = tmp_path / "predictions.jsonl"
    ids = (
        [json.loads(x)["request_id"] for x in audit.read_text().splitlines()]
        if audit.exists()
        else []
    )
    assert "late-1" not in ids, "audit claims a prediction the client never received"
    m = c.get("/metrics").text
    assert 'mlpipe_abandoned_inference_total{outcome="completed"} 1.0' in m


# R5 -----------------------------------------------------------------------------------
def test_r5_second_server_on_same_registry_refuses_to_start(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    a = TestClient(create_app(settings_factory()), headers=AUTH)
    a.__enter__()
    try:
        b = TestClient(create_app(settings_factory()), headers=AUTH)
        with pytest.raises(RuntimeError, match="serving lock"):
            b.__enter__()
    finally:
        a.__exit__(None, None, None)
    c = TestClient(create_app(settings_factory()), headers=AUTH)  # released on shutdown
    c.__enter__()
    assert c.get("/readyz").status_code == 200
    c.__exit__(None, None, None)


# R6 -----------------------------------------------------------------------------------
def test_r6_manifest_schema_and_fingerprint_must_agree(
    tmp_path: Path, artifacts: dict[str, Path]
) -> None:
    d = tmp_path / "v2-rf"
    shutil.copytree(artifacts["v2-rf"], d)
    m = json.loads((d / "manifest.json").read_text())
    m["feature_schema"]["version"] = "gdm-features/9"  # fingerprint left claiming v1
    (d / "manifest.json").write_text(json.dumps(m))
    rep = validate_artifact(d, SERVING_SCHEMA)
    assert not rep.passed and rep.failure is not None and rep.failure.name == "metadata"


# R7 -----------------------------------------------------------------------------------
def test_r7_known_bad_bytes_cannot_return_under_a_new_version(
    bootstrapped: Registry, settings_factory: Callable[..., Settings], tmp_path: Path
) -> None:
    ctl = _ctl(bootstrapped, settings_factory())
    ctl.promote("v2-rf")
    ctl.rollback(reason="bad")
    again = tmp_path / "v2-rf-again"
    shutil.copytree(bootstrapped.artifact_dir("v2-rf"), again)
    m = json.loads((again / "manifest.json").read_text())
    m["model_version"] = "v2-rf-again"
    (again / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RegistryError, match="known_bad_artifact"):
        bootstrapped.register(again)


# R8 -----------------------------------------------------------------------------------
def test_r8_orphaned_artifact_dir_from_crashed_register_is_quarantined(
    registry: Registry, artifacts: dict[str, Path], tmp_path: Path
) -> None:
    src = tmp_path / "v9"
    shutil.copytree(artifacts["v2-rf"], src)
    m = json.loads((src / "manifest.json").read_text())
    m["model_version"] = "v9"
    (src / "manifest.json").write_text(json.dumps(m))
    shutil.copytree(src, registry.models_dir / "v9")  # copied, but state never committed
    assert registry.register(src) == "v9"
    assert any(p.name.startswith(".orphan-v9-") for p in registry.models_dir.iterdir())
    assert registry.events()[-1].get("orphan_quarantined")


# R9 -----------------------------------------------------------------------------------
def test_r9_symlinked_artifact_directory_is_refused(
    registry: Registry, artifacts: dict[str, Path], tmp_path: Path
) -> None:
    d = registry.artifact_dir("v2-rf")
    outside = tmp_path / "elsewhere"
    shutil.move(d, outside)
    d.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RegistryError, match="artifact_path"):
        registry.artifact_dir("v2-rf")


# R10 ----------------------------------------------------------------------------------
def test_r10_promote_commit_that_lands_despite_an_error_activates_consistently(
    bootstrapped: Registry,
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctl = _ctl(bootstrapped, settings_factory())
    real = bootstrapped._commit

    def commit_then_fail(state: dict[str, Any]) -> None:
        real(state)
        if (state.get("probation") or {}).get("version") == "v2-rf":
            raise OSError("fsync of directory failed after rename")

    monkeypatch.setattr(bootstrapped, "_commit", commit_then_fail)
    res = ctl.promote("v2-rf")
    cur = ctl.current()
    # disk says v2 is active on probation; memory must agree (not keep serving v1)
    assert bootstrapped.read_state()["active"] == "v2-rf"
    assert res["result"] == "active_on_probation"
    assert cur is not None and cur.version == "v2-rf" and ctl.on_probation


# R11 ----------------------------------------------------------------------------------
def test_r11_candidate_left_by_crashed_promotion_is_released_on_restart(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    bootstrapped.transition("v2-rf", State.VALIDATED, action="validate")
    bootstrapped.transition("v2-rf", State.CANDIDATE, action="select_candidate")
    ctl = _ctl(bootstrapped, settings_factory())  # restart
    assert bootstrapped.state_of("v2-rf") == State.VALIDATED
    assert ctl.promote("v2-rf")["result"] == "active_on_probation"


# R12 ----------------------------------------------------------------------------------
def test_r12_registry_refuses_writes_that_break_the_single_active_invariant(
    bootstrapped: Registry,
) -> None:
    with pytest.raises(RegistryError, match=r"invalid_transition|invariant_violation"):
        bootstrapped.apply_many(
            [("v1-legacy-rf", State.RETIRED)],
            action="x",
            pointer_updates={"active": "v1-legacy-rf"},
        )
    with pytest.raises(RegistryError, match="invariant_violation"):
        bootstrapped.apply_many(
            [("v1-legacy-rf", State.ACTIVE)],
            action="x",
            pointer_updates={"active": None},
        )
