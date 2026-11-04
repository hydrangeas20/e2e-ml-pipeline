"""Health-gate policy (deterministic sequences; latencies are supplied, i.e. simulated
time) and request attribution under concurrency."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipe.config import Settings
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.deployment import (
    DeploymentController,
    DeploymentError,
    GateConfig,
    HealthGate,
)
from mlpipe.serving.observability import Metrics


def gate(**kw: Any) -> HealthGate:
    cfg = dict(
        min_samples=20,
        window=100,
        max_error_rate=0.10,
        max_consecutive_failures=5,
        max_p95_latency_ms=250.0,
    )
    cfg.update(kw)
    return HealthGate(GateConfig(**cfg))  # type: ignore[arg-type]


def feed(g: HealthGate, seq: str, latency: float = 10.0) -> list[str]:
    """seq: 'o' = success, 'x' = failure. Returns decisions."""
    return [g.record(c == "o", latency)[0] for c in seq]


# --- gate policy ---------------------------------------------------------------------


def test_error_rate_not_evaluated_below_min_samples() -> None:
    g = gate()
    assert feed(g, "xoxo" + "o" * 15) == ["observe"] * 19  # 2/19 = 10.5%, but n < 20


def test_error_rate_boundary_is_strictly_greater() -> None:
    g = gate()
    assert feed(g, "x" + "o" * 9 + "x" + "o" * 9)[-1] == "observe"  # 2/20 = 0.10, not > 0.10
    assert g.record(False, 10)[0] == "fail"  # 3/21 = 0.143


def test_consecutive_failures_trip_before_min_samples() -> None:
    assert feed(gate(), "xxxxx")[-1] == "fail"
    assert feed(gate(), "xxxxoxxxx") == ["observe"] * 9  # streak reset by a success


def test_p95_uses_linear_interpolation_over_successes_only() -> None:
    g = gate()
    for _ in range(19):
        g.record(True, 10.0)
    assert g.record(True, 1000.0)[0] == "observe"  # p95 = 10 + 0.05*990 = 59.5 ms
    g2 = gate()
    for _ in range(18):
        g2.record(True, 10.0)
    g2.record(True, 1000.0)
    decision, why = g2.record(True, 1000.0)  # 2/20 slow -> p95 = 1000 ms
    assert decision == "fail" and "p95 latency 1000.0" in why


def test_low_traffic_never_passes_before_window() -> None:
    g = gate(window=100)
    assert set(feed(g, "o" * 99)) == {"observe"}
    assert g.record(True, 10)[0] == "pass"


def test_no_traffic_keeps_probation_and_blocks_next_promotion(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    c = DeploymentController(bootstrapped, SERVING_SCHEMA, settings_factory(), Metrics())
    c.startup()
    c.promote("v2-rf")
    time.sleep(0.05)  # no requests at all: no evidence either way
    assert c.on_probation
    with pytest.raises(DeploymentError, match="probation_in_progress"):
        c.promote("v3-rf")


def test_client_and_infrastructure_errors_do_not_feed_the_gate(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
) -> None:
    c = client_factory(max_concurrency=1, max_queue=0, gate_max_consecutive_failures=2)
    c.post("/admin/promote", json={"version": "v2-rf"})
    for _ in range(5):  # 422: client error
        assert c.post("/predict", json={"Glucose": "x"}).status_code == 422
    c.post("/admin/faults", json={"kind": "audit_failure"})
    for _ in range(5):  # 503: audit infrastructure
        assert c.post("/v1/predict", json={"instances": rows[:1]}).status_code == 503
    c.delete("/admin/faults")
    probation = c.get("/admin/deployment").json()["probation"]
    assert probation is not None and probation["observed"] == 0
    assert c.get("/admin/deployment").json()["active"] == "v2-rf"


def test_timeouts_count_as_candidate_failures(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
) -> None:
    c = client_factory(
        max_concurrency=2, max_queue=4, request_timeout_s=0.05, gate_max_consecutive_failures=2
    )
    c.post("/admin/promote", json={"version": "v2-rf"})
    c.post("/admin/faults", json={"kind": "slow", "model_version": "v2-rf", "seconds": 0.3})
    assert [c.post("/v1/predict", json={"instances": rows[:1]}).status_code for _ in range(2)] == [
        504,
        504,
    ]
    assert c.get("/admin/deployment").json()["active"] == "v1-legacy-rf"


def test_failed_recovery_smoke_is_recorded_not_hidden(
    bootstrapped: Registry,
    settings_factory: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = DeploymentController(bootstrapped, SERVING_SCHEMA, settings_factory(), Metrics())
    c.startup()
    c.promote("v2-rf")
    monkeypatch.setattr(c, "_verify_recovery", lambda lm: False)
    assert c.rollback(reason="x")["result"] == "rolled_back"
    assert bootstrapped.events()[-1]["recovery_result"] == "smoke_failed"


@pytest.mark.parametrize("version", ["bad-corrupted", "v2-rf"])
def test_rejected_or_rolled_back_versions_cannot_be_reactivated(
    bootstrapped: Registry, settings_factory: Callable[..., Settings], version: str
) -> None:
    c = DeploymentController(bootstrapped, SERVING_SCHEMA, settings_factory(), Metrics())
    c.startup()
    if version == "v2-rf":
        c.promote("v2-rf")
        c.rollback(reason="bad")
    else:
        assert c.promote(version)["result"] == "rejected"
    with pytest.raises(DeploymentError, match="invalid_state"):
        c.promote(version)


# --- concurrency and attribution -------------------------------------------------------


def _hammer(
    c: TestClient,
    rows: list[dict[str, Any]],
    stop: threading.Event,
    out: list[tuple[int, str, str, float]],
) -> None:
    while not stop.is_set():
        r = c.post("/v1/predict", json={"instances": rows[:1]})
        b = r.json()
        out.append(
            (r.status_code, b["model_version"], b["request_id"], b["predictions"][0]["probability"])
        )


def _assert_attribution(seen: list[tuple[int, str, str, float]], audit_path: Path) -> None:
    by_version: dict[str, set[float]] = {}
    for _, v, _, p in seen:
        by_version.setdefault(v, set()).add(p)
    assert all(len(ps) == 1 for ps in by_version.values()), by_version
    audit = {
        json.loads(x)["request_id"]: json.loads(x)["model_version"]
        for x in audit_path.read_text().splitlines()
    }
    assert all(audit[rid] == v for _, v, rid, _ in seen)


def test_concurrent_predictions_during_rollback(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    c = client_factory(max_queue=64, gate_window=10_000)
    c.post("/admin/promote", json={"version": "v2-rf"})
    stop, seen = threading.Event(), []  # type: ignore[var-annotated]
    ts = [threading.Thread(target=_hammer, args=(c, rows, stop, seen)) for _ in range(4)]
    for t in ts:
        t.start()
    while len(seen) < 20:
        time.sleep(0.005)
    assert c.post("/admin/rollback", json={}).json()["result"] == "rolled_back"
    n = len(seen)
    while len(seen) < n + 20:
        time.sleep(0.005)
    stop.set()
    for t in ts:
        t.join()
    assert all(s == 200 for s, *_ in seen)
    assert {v for _, v, *_ in seen} == {"v1-legacy-rf", "v2-rf"}
    _assert_attribution(seen, tmp_path / "predictions.jsonl")


def test_concurrent_promotions_with_the_same_stale_counter(
    bootstrapped: Registry, settings_factory: Callable[..., Settings]
) -> None:
    c = DeploymentController(bootstrapped, SERVING_SCHEMA, settings_factory(), Metrics())
    c.startup()
    gen = bootstrapped.read_state()["generation"]
    barrier = threading.Barrier(4)
    results: list[str] = []

    def go(v: str) -> None:
        barrier.wait()
        try:
            results.append(c.promote(v, expected_generation=gen)["result"])
        except DeploymentError as e:
            results.append(e.code)

    ts = [threading.Thread(target=go, args=(v,)) for v in ("v2-rf", "v3-rf", "v2-rf", "v3-rf")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results.count("active_on_probation") == 1, results
    assert set(results) - {"active_on_probation"} <= {
        "promotion_in_progress",
        "stale_generation",
        "probation_in_progress",
    }
    with pytest.raises(DeploymentError, match="stale_generation"):
        c.promote("v3-rf", expected_generation=gen)


@pytest.mark.parametrize("attempt", range(5))  # fresh registry each; interleavings vary
def test_simultaneous_manual_and_automatic_rollback(
    bootstrapped: Registry, settings_factory: Callable[..., Settings], attempt: int
) -> None:
    c = DeploymentController(
        bootstrapped, SERVING_SCHEMA, settings_factory(gate_max_consecutive_failures=1), Metrics()
    )
    c.startup()
    c.promote("v2-rf")
    lm = c.current()
    assert lm is not None
    barrier = threading.Barrier(2)
    out: list[str] = []

    def manual() -> None:
        barrier.wait()
        out.append(c.rollback(reason="op")["result"])

    def auto() -> None:
        barrier.wait()
        c.record_outcome(lm, False, 1.0)

    ts = [threading.Thread(target=manual), threading.Thread(target=auto)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rb = [e for e in bootstrapped.events() if e["action"] == "rollback"]
    assert len(rb) == 1 and rb[0]["rollback_trigger"] in ("manual", "health_gate")
    assert out[0] == ("rolled_back" if rb[0]["rollback_trigger"] == "manual" else "noop")
    cur = c.current()
    assert cur is not None and cur.version == "v1-legacy-rf"


def test_request_spanning_a_swap_is_attributed_to_the_model_that_ran_it(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    c = client_factory(gate_max_consecutive_failures=1)
    c.post("/admin/faults", json={"kind": "slow", "model_version": "v1-legacy-rf", "seconds": 0.5})
    out: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: out.update(
            r=c.post(
                "/v1/predict", json={"instances": rows[:1]}, headers={"X-Request-ID": "spanning"}
            )
        )
    )
    t.start()
    time.sleep(0.1)
    assert c.post("/admin/promote", json={"version": "v2-rf"}).status_code == 200
    t.join()
    assert out["r"].status_code == 200 and out["r"].json()["model_version"] == "v1-legacy-rf"
    audit = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [a["model_version"] for a in audit if a["request_id"] == "spanning"] == ["v1-legacy-rf"]
    assert c.get("/admin/deployment").json()["probation"]["observed"] == 0


@pytest.mark.parametrize("late_outcome", ["success", "failure"])
def test_late_result_from_rolled_back_candidate_cannot_touch_next_gate(
    client_factory: Callable[..., TestClient],
    bootstrapped: Registry,
    rows: list[dict[str, Any]],
    tmp_path: Path,
    late_outcome: str,
) -> None:
    c = client_factory(gate_max_consecutive_failures=1, request_timeout_s=5)
    c.post("/admin/promote", json={"version": "v2-rf"})
    c.post("/admin/faults", json={"kind": "slow", "model_version": "v2-rf", "seconds": 0.6})
    if late_outcome == "failure":
        c.post("/admin/faults", json={"kind": "error_after_n", "model_version": "v2-rf", "n": 0})
    out: dict[str, Any] = {}
    t = threading.Thread(
        target=lambda: out.update(
            r=c.post("/v1/predict", json={"instances": rows[:1]}, headers={"X-Request-ID": "late"})
        )
    )
    t.start()
    time.sleep(0.1)
    assert c.post("/admin/rollback", json={}).json()["result"] == "rolled_back"
    assert c.post("/admin/promote", json={"version": "v3-rf"}).status_code == 200
    t.join()
    r = out["r"]
    assert (
        r.json().get(
            "model_version", r.json().get("error", {}).get("details", {}).get("model_version")
        )
        == "v2-rf"
    )  # attributed to the model that actually ran it
    dep = c.get("/admin/deployment").json()
    assert dep["active"] == "v3-rf" and dep["probation"]["observed"] == 0
    assert len([e for e in bootstrapped.events() if e["action"] == "rollback"]) == 1
