from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipe.registry import Registry

LEGACY_EXAMPLE = {  # the request example from the original README
    "Pregnancies": 4,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


def test_legacy_endpoints_preserved(client: TestClient) -> None:
    assert client.get("/").json() == {"status": "ok"}
    r = client.post("/predict", json=LEGACY_EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_version"] == "v1-legacy-rf"


def test_legacy_response_matches_original_implementation(client: TestClient) -> None:
    """Same prediction/confidence as the original `pickle.load` + `predict` code path."""
    import pickle

    import pandas as pd

    from tests.conftest import ROOT

    with (ROOT / "mlruns/0/models/m-ac8443cec0b64c03929f2d97d344b3c5/artifacts/model.pkl").open(
        "rb"
    ) as f:
        original = pickle.load(f)  # trusted: pinned owner artifact, test only
    df = pd.DataFrame([LEGACY_EXAMPLE])
    body = client.post("/predict", json=LEGACY_EXAMPLE).json()
    assert body["prediction"] == int(original.predict(df)[0])
    assert body["confidence"] == pytest.approx(float(original.predict_proba(df)[0].max()))


def test_v1_predict_contract_and_traceability(
    client: TestClient, rows: list[dict[str, Any]], tmp_path: Path
) -> None:
    r = client.post("/v1/predict", json={"instances": rows}, headers={"X-Request-ID": "abc-123"})
    assert r.status_code == 200
    b = r.json()
    assert r.headers["X-Request-ID"] == "abc-123" and b["request_id"] == "abc-123"
    assert b["model_version"] == "v1-legacy-rf" and len(b["model_sha256"]) == 64
    assert b["schema_version"] == "gdm-features/1" and b["config_version"]
    assert len(b["predictions"]) == 3
    for p in b["predictions"]:
        assert p["prediction"] == int(p["probability"] >= b["decision_threshold"])
    audit = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [a["request_id"] for a in audit] == ["abc-123"] * 3
    assert {a["model_sha256"] for a in audit} == {b["model_sha256"]}
    assert all("Glucose" not in a for a in audit), "no feature values in audit log"


def test_invalid_request_id_is_replaced(client: TestClient) -> None:
    r = client.get("/livez", headers={"X-Request-ID": "bad id with spaces\n"})
    assert r.headers["X-Request-ID"] != "bad id with spaces\n"
    assert len(r.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    "mutation,loc",
    [
        ({"Glucose": "148"}, "Glucose"),  # string
        ({"Glucose": True}, "Glucose"),  # bool
        ({"Glucose": 148.5}, "Glucose"),  # non-integer for int feature
        ({"Glucose": 9999}, "Glucose"),  # out of bounds
        ({"BMI": -1.0}, "BMI"),
        ({"Extra": 1}, "Extra"),  # unknown field
    ],
)
def test_strict_input_validation(client: TestClient, mutation: dict[str, Any], loc: str) -> None:
    r = client.post("/predict", json={**LEGACY_EXAMPLE, **mutation})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["request_id"]
    assert any(loc in e["loc"] for e in err["details"]["errors"])
    assert "9999" not in r.text and "148.5" not in r.text, "input values never echoed"


def test_missing_feature_and_nan_rejected(client: TestClient) -> None:
    payload = dict(LEGACY_EXAMPLE)
    del payload["Age"]
    assert client.post("/predict", json=payload).status_code == 422
    raw = json.dumps(LEGACY_EXAMPLE).replace("28.0", "NaN")
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch_limits(
    client_factory: Callable[..., TestClient], bootstrapped: Registry, rows: list[dict[str, Any]]
) -> None:
    c = client_factory(max_batch_rows=2)
    assert c.post("/v1/predict", json={"instances": []}).status_code == 422
    r = c.post("/v1/predict", json={"instances": rows})
    assert r.status_code == 422 and r.json()["error"]["code"] == "batch_too_large"


def test_not_ready_without_model(client_factory: Callable[..., TestClient]) -> None:
    c = client_factory()
    r = c.get("/readyz")
    assert r.status_code == 503 and "bootstrap" in r.json()["reason"]
    assert c.get("/livez").status_code == 200
    r = c.post("/predict", json=LEGACY_EXAMPLE)
    assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"


def test_readiness_fails_when_active_artifact_tampered(
    client: TestClient, bootstrapped: Registry
) -> None:
    assert client.get("/readyz").status_code == 200
    p = bootstrapped.artifact_dir("v1-legacy-rf") / "model.skops"
    b = bytearray(p.read_bytes())
    b[10] ^= 1
    p.write_bytes(bytes(b))
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["artifact_intact"] is False


def test_metrics_exposed(client: TestClient) -> None:
    client.post("/predict", json=LEGACY_EXAMPLE)
    client.post("/predict", json={**LEGACY_EXAMPLE, "Age": -5})
    text = client.get("/metrics").text
    for name in (
        "mlpipe_requests_total",
        "mlpipe_request_latency_seconds_bucket",
        "mlpipe_in_flight_requests",
        "mlpipe_active_model_info",
        "mlpipe_feature_validation_failures_total 1.0",
        'mlpipe_predictions_total{model_version="v1-legacy-rf"} 1.0',
        "mlpipe_audit_log_failures_total",
        "mlpipe_rollbacks_total",
        "mlpipe_model_load_failures_total",
    ):
        assert name in text, name


def test_admin_requires_token(
    client_factory: Callable[..., TestClient], bootstrapped: Registry
) -> None:
    c = client_factory()
    assert c.get("/admin/deployment", headers={"Authorization": "Bearer wrong"}).status_code == 401
    off = client_factory(admin_token=None)
    assert off.get("/admin/deployment").status_code == 403


def test_logs_are_json_and_omit_features(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    client.post("/predict", json=LEGACY_EXAMPLE, headers={"X-Request-ID": "log-check"})
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    req = [x for x in lines if x.get("request_id") == "log-check"]
    assert req and req[0]["route"] == "/predict" and req[0]["status"] == 200
    assert "148" not in json.dumps(req)
