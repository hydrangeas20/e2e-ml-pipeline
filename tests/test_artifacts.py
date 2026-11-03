from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from mlpipe.artifacts import (
    MODEL_FILE,
    ArtifactError,
    load_estimator,
    load_legacy_pickle,
    package_model,
    verify_integrity,
)
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.train import train_and_package
from mlpipe.validation import validate_artifact
from tests.conftest import DATA, ROOT


def test_legacy_import_preserves_logged_mlflow_metrics(artifacts: dict[str, Path]) -> None:
    m = json.loads((artifacts["v1-legacy-rf"] / "manifest.json").read_text())
    db = sqlite3.connect(ROOT / "mlflow.db")
    logged = dict(db.execute(
        "select key, value from metrics where run_uuid='dce997fa1be54b54a5013d93ef000865'"
    ).fetchall())
    for k in ("accuracy", "f1", "roc_auc", "pr_auc"):
        assert m["metrics"][k] == pytest.approx(logged[k], abs=1e-12), k
    assert m["provenance"]["legacy_md5"] == "333ad164dacb32f0d8844a27d7876c67"


def test_legacy_pickle_refused_unless_pinned(tmp_path: Path) -> None:
    fake = tmp_path / "model.pkl"
    fake.write_bytes(b"\x80\x04K\x01.")  # a valid pickle of the int 1
    with pytest.raises(ArtifactError, match="legacy_pin"):
        load_legacy_pickle(fake)


def test_manifest_records_provenance_and_schema(artifacts: dict[str, Path]) -> None:
    m = verify_integrity(artifacts["v2-rf"])
    assert m.feature_schema_fingerprint == SERVING_SCHEMA.fingerprint
    assert m.training["data_sha256"] and m.training["params"]["random_state"] == 42
    assert set(m.framework) == {"python", "sklearn", "skops"}
    assert m.metrics["roc_auc"] > 0.5


def test_training_is_deterministic(tmp_path: Path) -> None:
    a = train_and_package(DATA, tmp_path / "a", version="a", params={"n_estimators": 20})
    b = train_and_package(DATA, tmp_path / "b", version="b", params={"n_estimators": 20})
    assert a.metrics == b.metrics
    ea = load_estimator(tmp_path / "a", a)
    eb = load_estimator(tmp_path / "b", b)
    from mlpipe.features import transform
    from mlpipe.validation import smoke_frame

    X = transform(smoke_frame(), SERVING_SCHEMA)
    assert (ea.predict_proba(X) == eb.predict_proba(X)).all()


def test_tamper_detected_after_packaging(tmp_path: Path, artifacts: dict[str, Path]) -> None:
    import shutil

    d = tmp_path / "v2-rf"
    shutil.copytree(artifacts["v2-rf"], d)
    with (d / MODEL_FILE).open("r+b") as f:
        f.seek(100)
        f.write(b"\x00")
    with pytest.raises(ArtifactError, match="integrity"):
        verify_integrity(d)


def test_symlinked_model_refused(tmp_path: Path, artifacts: dict[str, Path]) -> None:
    d = tmp_path / "v2-rf"
    d.mkdir()
    (d / "manifest.json").write_bytes((artifacts["v2-rf"] / "manifest.json").read_bytes())
    os.symlink(artifacts["v2-rf"] / MODEL_FILE, d / MODEL_FILE)
    with pytest.raises(ArtifactError, match="model_missing"):
        verify_integrity(d)


def test_untrusted_types_are_refused_not_loaded(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(DATA)
    X, y = df[SERVING_SCHEMA.names], df["Outcome"]
    from sklearn.linear_model import LogisticRegression

    est = make_pipeline(FunctionTransformer(), LogisticRegression(max_iter=500)).fit(X, y)
    est.steps[0][1].func = os.getcwd  # smuggle a callable into a fitted, valid model
    m = package_model(est, tmp_path / "evil", model_version="evil", schema=SERVING_SCHEMA,
                      training={}, metrics={}, reference_stats={})
    with pytest.raises(ArtifactError, match="untrusted_types"):
        load_estimator(tmp_path / "evil", m)
    rep = validate_artifact(tmp_path / "evil", SERVING_SCHEMA)
    assert not rep.passed and rep.failure is not None and rep.failure.name == "load"
