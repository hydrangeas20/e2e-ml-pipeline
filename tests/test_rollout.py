"""Flagship: promotion, pre-promotion rejection, health-gated rollback, concurrency."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipe.config import Settings
from mlpipe.registry import Registry, State
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.deployment import DeploymentController, DeploymentError
from mlpipe.serving.observability import Metrics
from tests.test_validation_registry import EXPECTED_FAILURE


def _predict(c: TestClient, rows: list[dict[str, Any]]) -> tuple[int, str | None]:
    r = c.post("/v1/predict", json={"instances": rows})
    b = r.json()
    return r.status_code, b.get("model_version") or b.get("error", {}).get("details", {}).get(
        "model_version")


def _dep(c: TestClient) -> dict[str, Any]:
    return c.get("/admin/deployment").json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("version", sorted(EXPECTED_FAILURE))
def test_invalid_candidate_rejected_without_disrupting_active(
    client: TestClient, rows: list[dict[str, Any]], version: str
) -> None:
    before = _dep(client)
    r = client.post("/admin/promote", json={"version": version})
    assert r.status_code == 422 and r.json()["result"] == "rejected"
    assert r.json()["validation"]["failure_reason"].startswith(EXPECTED_FAILURE[version])
    after = _dep(client)
    assert after["active"] == before["active"] == "v1-legacy-rf"
    assert after["registry"]["versions"][version]["state"] == "rejected"
    assert _predict(client, rows) == (200, "v1-legacy-rf")
    # a rejected version can never be promoted later
    assert client.post("/admin/promote", json={"version": version}).status_code == 409


def test_full_rollout_failure_and_automatic_rollback(
    client: TestClient, rows: list[dict[str, Any]], bootstrapped: Registry
) -> None:
    assert _predict(client, rows) == (200, "v1-legacy-rf")
    r = client.post("/admin/promote", json={"version": "v2-rf"})
    assert r.json()["result"] == "active_on_probation"
    assert [_predict(client, rows) for _ in range(4)] == [(200, "v2-rf")] * 4
    client.post("/admin/faults", json={"kind": "error_after_n", "model_version": "v2-rf",
                                       "n": 2})
    trace = [_predict(client, rows) for _ in range(12)]
    # 2 more successes, then errors until the gate trips, then v1 serves again.
    assert trace[:2] == [(200, "v2-rf")] * 2
    first_v1 = trace.index((200, "v1-legacy-rf"))
    assert all(t == (500, "v2-rf") for t in trace[2:first_v1])
    assert all(t == (200, "v1-legacy-rf") for t in trace[first_v1:])
    d = _dep(client)
    assert d["active"] == "v1-legacy-rf" and d["probation"] is None
    assert d["registry"]["versions"]["v2-rf"]["state"] == "rolled_back"
    rb = [e for e in bootstrapped.events() if e["action"] == "rollback"]
    assert len(rb) == 1
    e = rb[0]
    assert e["rollback_trigger"] == "health_gate" and e["recovery_result"] == "verified"
    assert e["previous_active"] == "v2-rf" and e["next_active"] == "v1-legacy-rf"
    assert e["failure_reason"] and e["correlation_id"] and e["ts"] and e["gate"]["errors"] >= 1
    assert 'mlpipe_rollbacks_total{trigger="health_gate"} 1.0' in client.get("/metrics").text


def test_consecutive_failure_gate_trips_before_min_samples(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(gate_min_samples=50, gate_window=100, gate_max_consecutive_failures=3)
    c.post("/admin/promote", json={"version": "v2-rf"})
    c.post("/admin/faults", json={"kind": "error_after_n", "model_version": "v2-rf", "n": 0})
    trace = [_predict(c, rows) for _ in range(4)]
    assert trace == [(500, "v2-rf")] * 3 + [(200, "v1-legacy-rf")]


def test_probation_pass_makes_candidate_known_good(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(gate_min_samples=5, gate_window=10)
    c.post("/admin/promote", json={"version": "v2-rf"})
    for _ in range(10):
        assert _predict(c, rows) == (200, "v2-rf")
    d = _dep(c)
    assert d["known_good"] == "v2-rf" and d["probation"] is None
    st = bootstrapped.read_state()
    assert st["known_good"] == "v2-rf" and st["versions"]["v1-legacy-rf"]["state"] == "retired"
    assert c.post("/admin/rollback", json={}).json()["result"] == "noop"
    # next promotion is now allowed, and its known-good is v2
    r = c.post("/admin/promote", json={"version": "v3-rf"}).json()
    assert r["result"] == "active_on_probation" and r["known_good"] == "v2-rf"


def test_manual_rollback_and_idempotency(client: TestClient, rows: list[dict[str, Any]]) -> None:
    client.post("/admin/promote", json={"version": "v2-rf"})
    first = client.post("/admin/rollback", json={"reason": "operator"}).json()
    assert first["result"] == "rolled_back" and first["to"] == "v1-legacy-rf"
    for _ in range(3):
        assert client.post("/admin/rollback", json={}).json()["result"] == "noop"
    assert _predict(client, rows) == (200, "v1-legacy-rf")


def test_promotion_refused_during_probation_and_when_stale(client: TestClient) -> None:
    gen = _dep(client)["registry"]["generation"]
    r = client.post("/admin/promote", json={"version": "v2-rf", "expected_generation": gen + 5})
    assert r.status_code == 409 and r.json()["error"]["code"] == "stale_generation"
    assert client.post("/admin/promote", json={"version": "v2-rf"}).status_code == 200
    r = client.post("/admin/promote", json={"version": "v3-rf"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "probation_in_progress"


def _controller(reg: Registry, settings: Settings) -> DeploymentController:
    c = DeploymentController(reg, SERVING_SCHEMA, settings, Metrics())
    c.startup()
    return c


def test_concurrent_promotion_requests(bootstrapped: Registry,
                                       settings_factory: Callable[..., Settings]) -> None:
    ctl = _controller(bootstrapped, settings_factory())
    barrier = threading.Barrier(6)
    results: list[str] = []
    lock = threading.Lock()

    def go(v: str) -> None:
        barrier.wait()
        try:
            out = ctl.promote(v)["result"]
        except DeploymentError as e:
            out = e.code
        with lock:
            results.append(out)

    ts = [threading.Thread(target=go, args=(v,)) for v in ["v2-rf", "v3-rf"] * 3]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results.count("active_on_probation") == 1, results
    assert set(results) <= {"active_on_probation", "promotion_in_progress",
                            "probation_in_progress", "invalid_state"}
    st = bootstrapped.read_state()
    assert [v for v, e in st["versions"].items() if e["state"] == "active"] == [st["active"]]


def test_concurrent_inference_during_promotion_is_consistently_attributed(
    client_factory: Callable[..., TestClient], bootstrapped: Registry,
    rows: list[dict[str, Any]], tmp_path: Path,
) -> None:
    c = client_factory(max_concurrency=4, max_queue=64, gate_window=10_000)
    stop = threading.Event()
    seen: list[tuple[int, str, str, list[float]]] = []

    def hammer() -> None:
        while not stop.is_set():
            r = c.post("/v1/predict", json={"instances": rows[:1]})
            b = r.json()
            seen.append((r.status_code, b["model_version"], b["request_id"],
                         [p["probability"] for p in b["predictions"]]))

    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(hammer) for _ in range(4)]
        while len(seen) < 20:
            pass
        assert c.post("/admin/promote", json={"version": "v2-rf"}).status_code == 200
        n0 = len(seen)
        while len(seen) < n0 + 20:
            pass
        stop.set()
        for f in futs:
            f.result()
    assert all(s == 200 for s, *_ in seen)
    assert {v for _, v, *_ in seen} == {"v1-legacy-rf", "v2-rf"}
    # each version always gives the same probability for the same row: no mixing
    by_v: dict[str, set[float]] = {}
    for _, v, _, p in seen:
        by_v.setdefault(v, set()).add(p[0])
    assert all(len(s) == 1 for s in by_v.values()) and len(by_v) == 2
    # the audit record of every request names the same version as its response
    audit = {json.loads(x)["request_id"]: json.loads(x)["model_version"]
             for x in (tmp_path / "predictions.jsonl").read_text().splitlines()}
    assert all(audit[rid] == v for _, v, rid, _ in seen)


def test_interrupted_promotion_keeps_active_model(
    bootstrapped: Registry, settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctl = _controller(bootstrapped, settings_factory())
    real = bootstrapped.apply_many

    def crash(*a: Any, **kw: Any) -> Any:
        if kw.get("action") == "promote":
            raise OSError("simulated crash during commit")
        return real(*a, **kw)

    monkeypatch.setattr(bootstrapped, "apply_many", crash)
    with pytest.raises(DeploymentError, match="promotion_failed"):
        ctl.promote("v2-rf")
    cur = ctl.current()
    assert cur is not None and cur.version == "v1-legacy-rf" and not ctl.on_probation
    assert bootstrapped.state_of("v2-rf") == State.VALIDATED  # put back, can retry
    monkeypatch.undo()
    assert ctl.promote("v2-rf")["result"] == "active_on_probation"


def test_failed_activation_artifact_modified_after_validation(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    ctl = _controller(bootstrapped, settings_factory())
    assert ctl.validate("v2-rf")["passed"]
    p = bootstrapped.artifact_dir("v2-rf") / "model.skops"
    b = bytearray(p.read_bytes())
    b[-5] ^= 1
    p.write_bytes(bytes(b))
    res = ctl.promote("v2-rf")
    assert res["result"] == "rejected" and "integrity" in res["validation"]["failure_reason"]
    cur = ctl.current()
    assert cur is not None and cur.version == "v1-legacy-rf"


def test_restart_during_probation_restores_known_good(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    ctl = _controller(bootstrapped, settings_factory())
    ctl.promote("v2-rf")  # process "dies" here, mid-probation
    fresh = _controller(bootstrapped, settings_factory())
    cur = fresh.current()
    assert cur is not None and cur.version == "v1-legacy-rf"
    st = bootstrapped.read_state()
    assert st["active"] == "v1-legacy-rf" and st["probation"] is None
    assert st["versions"]["v2-rf"]["state"] == "rolled_back"
    assert bootstrapped.events()[-1]["rollback_trigger"] == "restart_during_probation"


def test_restart_with_unloadable_known_good_is_not_ready(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    (bootstrapped.artifact_dir("v1-legacy-rf") / "model.skops").write_bytes(b"garbage")
    ctl = _controller(bootstrapped, settings_factory())
    assert ctl.current() is None
    r = ctl.readiness()
    assert not r["ready"] and "failed to load" in str(r["reason"])


def test_late_outcomes_from_replaced_model_are_ignored(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    ctl = _controller(bootstrapped, settings_factory(gate_max_consecutive_failures=1))
    old = ctl.current()
    assert old is not None
    ctl.promote("v2-rf")
    ctl.record_outcome(old, False, 1.0)  # stale failure attributed to v1, not v2
    cur = ctl.current()
    assert cur is not None and cur.version == "v2-rf" and ctl.on_probation
