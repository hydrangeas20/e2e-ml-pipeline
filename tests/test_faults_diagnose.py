"""Additional fault injection (section 7) and the model-quality diagnosis workflow."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlpipe.artifacts import load_estimator, verify_integrity
from mlpipe.config import Settings
from mlpipe.diagnose import diagnose
from mlpipe.features import transform
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.validation import smoke_frame
from tests.conftest import DATA

# -- feature transformation defect ----------------------------------------------------


def test_feature_defect_detected_by_proxy_monitor(
    client: TestClient, rows: list[dict[str, Any]]
) -> None:
    batch = smoke_frame().to_dict(orient="records")
    client.post("/admin/faults", json={"kind": "feature_defect", "defect": "glucose_mmol"})
    for _ in range(25):
        assert client.post("/v1/predict", json={"instances": batch}).status_code == 200
    mon = client.get("/admin/deployment").json()["proxy_monitoring"]
    assert mon["alert_features"] == ["Glucose"]
    assert "not a model-quality measurement" in mon["note"]
    assert 'mlpipe_feature_mean_shift{feature="Glucose"}' in client.get("/metrics").text
    # recovery: fix (clear) the defect; the monitor returns to normal on fresh traffic
    client.delete("/admin/faults")
    for _ in range(25):  # 800 clean rows: the 500-row window holds only post-fix traffic
        client.post("/v1/predict", json={"instances": batch})
    assert client.get("/admin/deployment").json()["proxy_monitoring"]["alert_features"] == []


def test_regression_guard_serving_transform_is_identity_on_schema_columns() -> None:
    """The production transform must equal the raw values (no hidden rescaling)."""
    raw = pd.read_csv(DATA)
    X = transform(raw, SERVING_SCHEMA)
    assert list(X.columns) == SERVING_SCHEMA.names
    assert (X.to_numpy() == raw[SERVING_SCHEMA.names].to_numpy(dtype="float64")).all()


# -- overload, timeout ----------------------------------------------------------------


def test_overload_is_refused_at_admission(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(max_concurrency=1, max_queue=1, request_timeout_s=5)
    c.post("/admin/faults", json={"kind": "slow", "seconds": 0.5})
    codes: list[int] = []
    lock = threading.Lock()

    def go() -> None:
        r = c.post("/v1/predict", json={"instances": rows[:1]})
        with lock:
            codes.append(r.status_code)

    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert codes.count(200) == 2 and codes.count(429) == 4, codes
    assert 'mlpipe_admission_rejected_total{reason="overloaded"} 4.0' in c.get("/metrics").text
    c.delete("/admin/faults")
    assert c.post("/v1/predict", json={"instances": rows[:1]}).status_code == 200


def test_timeout_returns_504_and_keeps_slot_until_work_finishes(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(max_concurrency=1, max_queue=0, request_timeout_s=0.1)
    c.post("/admin/faults", json={"kind": "slow", "seconds": 0.6})
    r = c.post("/v1/predict", json={"instances": rows[:1]})
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
    # thread still busy: capacity is genuinely bounded (documented limitation)
    assert c.post("/v1/predict", json={"instances": rows[:1]}).status_code == 429
    c.delete("/admin/faults")
    time.sleep(0.7)
    assert c.post("/v1/predict", json={"instances": rows[:1]}).status_code == 200


# -- audit log failure -----------------------------------------------------------------


def test_required_audit_failure_withholds_prediction(
    client: TestClient, rows: list[dict[str, Any]], tmp_path: Path
) -> None:
    client.post("/v1/predict", json={"instances": rows[:1]})
    n_before = len((tmp_path / "predictions.jsonl").read_text().splitlines())
    client.post("/admin/faults", json={"kind": "audit_failure"})
    r = client.post("/v1/predict", json={"instances": rows[:1]})
    assert r.status_code == 503 and r.json()["error"]["code"] == "audit_unavailable"
    assert "predictions" not in r.json()
    assert "mlpipe_audit_log_failures_total 1.0" in client.get("/metrics").text
    # audit failures are infrastructure, not a candidate defect: no rollback trigger
    assert client.get("/admin/deployment").json()["active"] == "v1-legacy-rf"
    client.delete("/admin/faults")
    assert client.post("/v1/predict", json={"instances": rows[:1]}).status_code == 200
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == n_before + 1


def test_optional_audit_failure_still_serves(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(audit_required=False)
    c.post("/admin/faults", json={"kind": "audit_failure"})
    assert c.post("/v1/predict", json={"instances": rows[:1]}).status_code == 200
    assert "mlpipe_audit_log_failures_total 1.0" in c.get("/metrics").text


def test_audit_write_is_all_or_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from mlpipe.serving import audit as audit_mod
    from mlpipe.serving.audit import AuditError, PredictionAuditLog

    log = PredictionAuditLog(tmp_path / "a.jsonl", fsync=True, fail_hook=lambda: False)
    log.write([{"x": 1}])
    real_write = os.write
    monkeypatch.setattr(audit_mod.os, "write", lambda fd, b: real_write(fd, b[:5]))
    with pytest.raises(AuditError):
        log.write([{"x": 2}, {"x": 3}])
    monkeypatch.undo()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["x"] == 1


# -- invalid configuration ----------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"MLPIPE_MAX_CONCURRENCY": "0"},
        {"MLPIPE_GATE_MIN_SAMPLES": "50", "MLPIPE_GATE_WINDOW": "10"},
        {"MLPIPE_REQUEST_TIMEOUT_S": "-1"},
        {"MLPIPE_ADMIN_TOKEN": "short"},
        {"MLPIPE_LOG_LEVEL": "LOUD"},
        {"MLPIPE_AUDIT_REQUIRED": "maybe"},
    ],
)
def test_invalid_configuration_fails_fast(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError):
        Settings()


def test_config_version_excludes_secret_but_tracks_settings() -> None:
    a = Settings(admin_token="a" * 20)
    b = Settings(admin_token="b" * 20)
    c = Settings(admin_token="a" * 20, gate_window=500)
    assert a.config_version == b.config_version != c.config_version
    assert "a" * 20 not in json.dumps(a.model_dump(mode="json"))


def test_faults_disabled_by_default(
    client_factory: Callable[..., TestClient], bootstrapped: Registry
) -> None:
    c = client_factory(enable_fault_injection=False)
    r = c.post("/admin/faults", json={"kind": "audit_failure"})
    assert r.status_code == 403


# -- graceful shutdown -----------------------------------------------------------------


def test_graceful_shutdown_drains_in_flight_requests(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(drain_timeout_s=5)
    c.post("/admin/faults", json={"kind": "slow", "seconds": 0.5})
    out: dict[str, int] = {}

    def slow_request() -> None:
        out["code"] = c.post("/v1/predict", json={"instances": rows[:1]}).status_code

    t = threading.Thread(target=slow_request)
    t.start()
    time.sleep(0.15)
    lifecycle = c.app.state.lifecycle  # type: ignore[attr-defined]
    assert lifecycle["admitted"] == 1
    t0 = time.monotonic()
    c.__exit__(None, None, None)  # runs lifespan shutdown: drain
    t.join()
    assert out["code"] == 200, "in-flight request completed during drain"
    assert lifecycle["draining"] and lifecycle["admitted"] == 0
    assert time.monotonic() - t0 >= 0.2


# -- diagnosis -------------------------------------------------------------------------


@pytest.fixture
def legacy(artifacts: dict[str, Path]) -> tuple[Any, dict[str, Any]]:
    m = verify_integrity(artifacts["v1-legacy-rf"])
    return load_estimator(artifacts["v1-legacy-rf"], m), m.model_dump()


def test_diagnose_clean(legacy: tuple[Any, dict[str, Any]]) -> None:
    rep = diagnose(*legacy, pd.read_csv(DATA))
    assert rep["likely_root_cause"] == "none_detected", rep


def test_diagnose_broken_feature_transform(legacy: tuple[Any, dict[str, Any]]) -> None:
    rep = diagnose(*legacy, pd.read_csv(DATA), defect="glucose_mmol")
    assert rep["likely_root_cause"] == "feature_pipeline"
    fp = rep["layers"]["feature_pipeline"]
    assert fp["features_differing_from_reference_transform"] == ["Glucose"]
    mq = rep["layers"]["model_quality"]
    # downstream symptom is visible but not blamed: model is fine on correct features
    assert mq["roc_auc_served"] < mq["roc_auc_reference_transform"] - 0.03
    assert mq["status"] == "ok"


def test_diagnose_data_quality(legacy: tuple[Any, dict[str, Any]]) -> None:
    raw = pd.read_csv(DATA)
    raw.loc[raw.index[:60], "BMI"] = 0  # upstream starts sending missing-as-zero
    rep = diagnose(*legacy, raw)
    assert rep["likely_root_cause"] == "data_quality"
    assert any("BMI" in i for i in rep["layers"]["data_quality"]["issues"])


def test_diagnose_threshold_problem(legacy: tuple[Any, dict[str, Any]]) -> None:
    est, m = legacy
    rep = diagnose(est, {**m, "decision_threshold": 0.95}, pd.read_csv(DATA))
    assert rep["likely_root_cause"] == "calibration"
    assert rep["layers"]["model_quality"]["status"] == "ok"


def test_diagnose_model_quality(legacy: tuple[Any, dict[str, Any]]) -> None:
    # concept drift: inputs unchanged, but the feature->outcome relation changed for
    # a deterministic 35% of rows (labels flipped). Only labels can reveal this.
    raw = pd.read_csv(DATA)
    flip = raw.sample(frac=0.35, random_state=0).index
    raw.loc[flip, "Outcome"] = 1 - raw.loc[flip, "Outcome"]
    rep = diagnose(*legacy, raw)
    assert rep["layers"]["data_quality"]["status"] == "ok"
    assert rep["layers"]["feature_pipeline"]["status"] == "ok"
    assert rep["likely_root_cause"] == "model_quality"
    unlabeled = diagnose(*legacy, raw.drop(columns=["Outcome"]), labels=False)
    assert unlabeled["likely_root_cause"] == "none_detected"  # invisible to proxies


def test_diagnose_without_labels_never_claims_model_quality(
    legacy: tuple[Any, dict[str, Any]],
) -> None:
    rep = diagnose(*legacy, pd.read_csv(DATA).drop(columns=["Outcome"]), labels=False)
    assert rep["layers"]["model_quality"]["status"] == "unknown"
    assert rep["layers"]["calibration"]["status"] == "unknown"


def test_diagnose_serving_layer(legacy: tuple[Any, dict[str, Any]]) -> None:
    rep = diagnose(*legacy, pd.read_csv(DATA), server_metrics={"timeouts": 12, "errors": 0})
    assert rep["likely_root_cause"] == "serving"
