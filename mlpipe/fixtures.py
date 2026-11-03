"""Deterministic bad-candidate fixtures for pre-promotion rejection tests and the demo.

Each builder writes a complete artifact directory whose defect is known by
construction:

* ``corrupted``       valid artifact, then one byte of model.skops flipped (hash mismatch)
* ``schema_mismatch`` manifest declares a different feature schema version
* ``lying_schema``    manifest claims the serving schema, model trained without Insulin
* ``nan_output``      loads fine, predict_proba returns NaN
* ``load_failure``    hash-consistent garbage bytes: integrity passes, deserialization fails
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from mlpipe.artifacts import MANIFEST_FILE, MODEL_FILE, Manifest, package_model, sha256_file
from mlpipe.features import transform
from mlpipe.schema import SERVING_SCHEMA, TARGET, FeatureSchema, FeatureSpec
from mlpipe.train import train_and_package

FIXTURES = ("corrupted", "schema_mismatch", "lying_schema", "nan_output", "load_failure")


def _rewrite_manifest(d: Path, **updates: object) -> None:
    m = json.loads((d / MANIFEST_FILE).read_text())
    m.update(updates)
    (d / MANIFEST_FILE).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def build_fixture(kind: str, data_csv: Path, out_root: Path, prefix: str = "bad") -> Path:
    version = f"{prefix}-{kind.replace('_', '-')}"
    d = out_root / version
    small = {"n_estimators": 10, "max_depth": 4}
    if kind == "corrupted":
        train_and_package(data_csv, d, version=version, params=small)
        p = d / MODEL_FILE
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0xFF
        p.write_bytes(bytes(b))
    elif kind == "schema_mismatch":
        other = FeatureSchema(
            version="gdm-features/2",
            features=(*SERVING_SCHEMA.features, FeatureSpec("HbA1c", "float", 0, 20)),
        )
        train_and_package(data_csv, d, version=version, params=small)
        _rewrite_manifest(d, feature_schema=other.to_dict(), feature_schema_fingerprint=other.fingerprint)
    elif kind == "lying_schema":
        train_and_package(data_csv, d, version=version, params=small, drop_features=["Insulin"])
    elif kind == "nan_output":
        df = pd.read_csv(data_csv)
        est = DummyClassifier(strategy="prior").fit(transform(df, SERVING_SCHEMA), df[TARGET])
        est.class_prior_ = np.array([np.nan, np.nan])
        package_model(
            est,
            d,
            model_version=version,
            schema=SERVING_SCHEMA,
            training={"fixture": kind},
            metrics={},
            reference_stats={},
        )
    elif kind == "load_failure":
        train_and_package(data_csv, d, version=version, params=small)
        p = d / MODEL_FILE
        p.write_bytes(b"PK\x03\x04 this is not a skops archive " * 8)
        _rewrite_manifest(d, model_sha256=sha256_file(p), model_bytes=p.stat().st_size)
    else:
        raise ValueError(f"unknown fixture {kind!r}; choose from {FIXTURES}")
    Manifest.model_validate_json((d / MANIFEST_FILE).read_text())  # still parseable
    return d
