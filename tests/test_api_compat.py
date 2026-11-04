"""Legacy /predict vs /v1/predict input rules (owner decision, 2026-11-03).

The two routes share one schema and one set of bounds. The only difference: the
deprecated /predict accepts a finite float with no fractional part for an integer
feature and converts it (148.0 -> 148). /v1/predict stays strictly typed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

BASE: dict[str, Any] = {
    "Pregnancies": 4,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}

# (case id, raw JSON fragment for Glucose or a payload transform, legacy status, v1 status)
CASES = [
    ("int", {"Glucose": 148}, 200, 200),
    ("exact_float", {"Glucose": 148.0}, 200, 422),
    ("exponent_float", {"Glucose": 1.48e2}, 200, 422),
    ("zero_float", {"Insulin": 0.0}, 200, 422),
    ("fractional_float", {"Glucose": 148.5}, 422, 422),
    ("tiny_fraction", {"Glucose": 148.000001}, 422, 422),
    ("numeric_string", {"Glucose": "148"}, 422, 422),
    ("float_string", {"Glucose": "148.0"}, 422, 422),
    ("bool", {"Glucose": True}, 422, 422),
    ("null", {"Glucose": None}, 422, 422),
    ("out_of_range_int", {"Glucose": 401}, 422, 422),
    ("out_of_range_exact_float", {"Glucose": 401.0}, 422, 422),
    ("negative_exact_float", {"Age": -1.0}, 422, 422),
    ("huge_exact_float", {"Glucose": 1e300}, 422, 422),
    ("upper_bound_exact_float", {"Glucose": 400.0}, 200, 422),
]


def _legacy(c: TestClient, payload: dict[str, Any]) -> int:
    return c.post("/predict", json=payload).status_code


def _v1(c: TestClient, payload: dict[str, Any]) -> int:
    return c.post("/v1/predict", json={"instances": [payload]}).status_code


@pytest.mark.parametrize("name,mutation,legacy,v1", CASES, ids=[c[0] for c in CASES])
def test_route_input_rules(
    client: TestClient, name: str, mutation: dict[str, Any], legacy: int, v1: int
) -> None:
    payload = {**BASE, **mutation}
    assert _legacy(client, payload) == legacy
    assert _v1(client, payload) == v1


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_rejected_on_both_routes(client: TestClient, literal: str) -> None:
    for field in ("Glucose", "BMI"):  # an integer and a float feature
        raw = json.dumps({**BASE, field: "__X__"}).replace('"__X__"', literal)
        r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, (field, literal)
        v1 = '{"instances": [' + raw + "]}"
        r = client.post("/v1/predict", content=v1, headers={"content-type": "application/json"})
        assert r.status_code == 422, (field, literal)


@pytest.mark.parametrize("change", ["unknown", "missing"])
def test_unknown_or_missing_features_rejected_on_both_routes(
    client: TestClient, change: str
) -> None:
    payload = dict(BASE)
    if change == "unknown":
        payload["Extra"] = 1
    else:
        del payload["Age"]
    assert _legacy(client, payload) == 422
    assert _v1(client, payload) == 422


def test_exact_float_is_served_identically_to_the_integer(client: TestClient) -> None:
    as_int = client.post("/predict", json=BASE).json()
    as_float = client.post("/predict", json={**BASE, "Glucose": 148.0, "Age": 50.0}).json()
    for k in ("prediction", "confidence", "probability", "model_version"):
        assert as_float[k] == as_int[k], k
    v1 = client.post("/v1/predict", json={"instances": [BASE]}).json()
    assert v1["predictions"][0]["probability"] == as_int["probability"]


def test_legacy_route_marked_deprecated_in_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert paths["/predict"]["post"].get("deprecated") is True
    assert not paths["/v1/predict"]["post"].get("deprecated", False)
