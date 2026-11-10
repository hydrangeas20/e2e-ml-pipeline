from __future__ import annotations

import threading
from pathlib import Path

import pytest

from mlpipe import registry as registry_mod
from mlpipe.registry import Registry, RegistryError, StaleGenerationError, State
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.validation import validate_artifact

EXPECTED_FAILURE = {
    "bad-corrupted": "integrity",
    "bad-schema-mismatch": "schema_compatibility",
    "bad-lying-schema": "model_features",
    "bad-nan-output": "numerical_validity",
    "bad-load-failure": "load",
}


@pytest.mark.parametrize("version,check", sorted(EXPECTED_FAILURE.items()))
def test_each_fixture_rejected_by_intended_check(
    artifacts: dict[str, Path], version: str, check: str
) -> None:
    rep = validate_artifact(artifacts[version], SERVING_SCHEMA)
    assert not rep.passed
    assert rep.failure is not None and rep.failure.name == check
    assert rep.estimator is None


@pytest.mark.parametrize("version", ["v1-legacy-rf", "v2-rf", "v3-rf"])
def test_good_artifacts_pass_every_check(artifacts: dict[str, Path], version: str) -> None:
    rep = validate_artifact(artifacts[version], SERVING_SCHEMA)
    assert rep.passed, rep.summary()
    assert [c.name for c in rep.checks] == [
        "integrity",
        "metadata",
        "schema_compatibility",
        "load",
        "model_features",
        "prediction_contract",
        "numerical_validity",
        "resources",
    ]


def test_corrupted_artifact_is_never_deserialized(
    artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlpipe.validation as v

    called = []
    monkeypatch.setattr(v, "load_estimator", lambda *a: called.append(a))
    validate_artifact(artifacts["bad-corrupted"], SERVING_SCHEMA)
    assert called == []


def test_state_machine_rejects_invalid_transitions(registry: Registry) -> None:
    registry.transition("v2-rf", State.VALIDATED, action="validate")
    with pytest.raises(RegistryError, match="invalid_transition"):
        registry.transition("v2-rf", State.ROLLED_BACK, action="x")
    registry.transition("bad-corrupted", State.REJECTED, action="validate")
    for to in State:
        with pytest.raises(RegistryError):
            registry.transition("bad-corrupted", to, action="x")


def test_stale_generation_refused(registry: Registry) -> None:
    gen = registry.read_state()["generation"]
    with pytest.raises(StaleGenerationError):
        registry.transition(
            "v2-rf", State.VALIDATED, action="validate", expected_generation=gen - 1
        )
    registry.transition("v2-rf", State.VALIDATED, action="validate", expected_generation=gen)


def test_duplicate_registration_refused(registry: Registry, artifacts: dict[str, Path]) -> None:
    with pytest.raises(RegistryError, match="duplicate_version"):
        registry.register(artifacts["v2-rf"])


def test_interrupted_commit_leaves_state_unchanged_and_reports_uncommitted(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = registry.read_state()

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(registry_mod, "write_json_atomic", boom)
    with pytest.raises(OSError):
        registry.transition("v2-rf", State.VALIDATED, action="validate")
    monkeypatch.undo()
    assert registry.read_state() == before
    unc = registry.inspect()["uncommitted_events"]
    assert len(unc) == 1 and unc[0]["model_version"] == "v2-rf"


def test_concurrent_transitions_serialize(registry: Registry) -> None:
    errors: list[Exception] = []

    def go() -> None:
        try:
            registry.transition("v2-rf", State.VALIDATED, action="validate")
        except RegistryError as e:
            errors.append(e)

    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(errors) == 7  # exactly one REGISTERED -> VALIDATED succeeded
    gens = [e["generation"] for e in registry.events()]
    assert gens == sorted(set(gens)), "generations unique and monotonic"


def test_state_survives_new_registry_instance(registry: Registry) -> None:
    registry.transition("v2-rf", State.VALIDATED, action="validate")
    again = Registry(registry.root)
    assert again.state_of("v2-rf") == State.VALIDATED
    assert again.events()[-1]["action"] == "validate"
