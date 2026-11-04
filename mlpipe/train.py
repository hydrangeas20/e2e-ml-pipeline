"""Deterministic training, evaluation and packaging of model artifacts.

This is the artifact-producing counterpart of the legacy ``src/train.py`` (which is
kept unchanged in behaviour for the DVC/MLflow workflow). Differences:

* fixed hyperparameters by default (grid search is opt-in), so a version is a
  deterministic function of (data sha256, params, library versions);
* the output is a versioned skops artifact with manifest, not a bare pickle.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlpipe.artifacts import LEGACY_PICKLE_PATH, Manifest, load_legacy_pickle, package_model
from mlpipe.features import reference_stats, transform
from mlpipe.schema import SERVING_SCHEMA, TARGET, FeatureSchema

# Public copy of the source dataset referenced in the README. It differs from the
# DVC-tracked data/raw/data.csv (not recoverable without DagsHub credentials) in one
# row; see docs/PLATFORM.md#data-provenance.
PUBLIC_DATA_URL = "https://raw.githubusercontent.com/maghfera/Diabetes-prediction/main/diabetes.csv"
PUBLIC_DATA_SHA256 = "bf81d06e0c0512662862248241ab46d7c29258c6ceaad32f3cab56b52a17fdab"

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def evaluate_binary(y: Any, proba: Any, threshold: float = 0.5) -> dict[str, float]:
    p1 = np.asarray(proba)[:, 1]
    pred = (p1 >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p1)),
        "pr_auc": float(average_precision_score(y, p1)),
        "brier": float(brier_score_loss(y, p1)),
        "positive_rate": float(pred.mean()),
    }


def split(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=df[TARGET]
    )
    return train, test


def train_and_package(
    data_csv: Path,
    out_dir: Path,
    *,
    version: str,
    params: dict[str, Any] | None = None,
    schema: FeatureSchema = SERVING_SCHEMA,
    drop_features: list[str] | None = None,
) -> Manifest:
    raw = data_csv.read_bytes()
    df = pd.read_csv(data_csv)
    p = {**DEFAULT_PARAMS, **(params or {})}
    train, test = split(df, int(p["random_state"]))
    X_tr = transform(train, schema)
    X_te = transform(test, schema)
    if drop_features:  # only used to build the incompatible-schema fixture
        X_tr = X_tr.drop(columns=drop_features)
        X_te = X_te.drop(columns=drop_features)
    model = RandomForestClassifier(n_jobs=1, **p).fit(X_tr, train[TARGET])
    metrics = evaluate_binary(test[TARGET], model.predict_proba(X_te))
    return package_model(
        model,
        out_dir,
        model_version=version,
        schema=schema,
        training={
            "data_path": str(data_csv),
            "data_sha256": sha256_bytes(raw),
            "n_train": len(train),
            "n_test": len(test),
            "split": "stratified 80/20",
            "params": p,
        },
        metrics=metrics,
        reference_stats=reference_stats(X_tr),
        provenance={"source": "mlpipe.train"},
    )


def import_legacy(
    out_dir: Path,
    *,
    version: str = "v1-legacy-rf",
    pickle_path: Path = LEGACY_PICKLE_PATH,
    holdout_csv: Path = Path("data/splits/test.csv"),
) -> Manifest:
    """Repackage the legacy pinned pickle as a skops artifact, preserving behaviour.

    Its training split is not recoverable (raw data lives only on the DVC remote), so
    reference stats are computed on the committed holdout split and labelled as such.
    """
    model = load_legacy_pickle(pickle_path)
    model.set_params(n_jobs=1)  # deterministic, bounded threads inside the server
    holdout = pd.read_csv(holdout_csv)
    X = transform(holdout, SERVING_SCHEMA)
    metrics = evaluate_binary(holdout[TARGET], model.predict_proba(X))
    return package_model(
        model,
        out_dir,
        model_version=version,
        schema=SERVING_SCHEMA,
        training={
            "data_path": "data/raw/data.csv (DVC, md5 739f3c9177b13d1c15aa060046cfb023)",
            "params": {k: v for k, v in model.get_params().items() if v is not None},
            "reference_stats_source": str(holdout_csv),
        },
        metrics=metrics,
        reference_stats=reference_stats(X),
        provenance={
            "source": "legacy pickle (git 37995fe)",
            "legacy_path": str(pickle_path),
            "legacy_md5": "333ad164dacb32f0d8844a27d7876c67",
            "mlflow_run_id": "dce997fa1be54b54a5013d93ef000865",
        },
    )
