"""Pre-promotion candidate validation.

Every check is recorded (name, passed, detail). Checks short-circuit after the
first failure that makes later checks meaningless (e.g. no load after an integrity
failure: never deserialize bytes that failed their hash).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlpipe.artifacts import ArtifactError, Manifest, load_estimator, verify_integrity
from mlpipe.features import transform
from mlpipe.schema import FeatureSchema


@dataclass(frozen=True)
class ResourceLimits:
    max_load_seconds: float = 10.0
    max_smoke_ms_per_row: float = 50.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    version: str
    checks: list[Check] = field(default_factory=list)
    manifest: Manifest | None = None
    estimator: Any = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failure(self) -> Check | None:
        return next((c for c in self.checks if not c.passed), None)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
            "failure_reason": f"{self.failure.name}: {self.failure.detail}"
            if self.failure
            else None,
        }


def smoke_frame() -> pd.DataFrame:
    with resources.files("mlpipe.data").joinpath("smoke.csv").open() as f:
        return pd.read_csv(f)


def check_outputs(proba: Any, n_rows: int) -> str | None:
    """Return an error string if predict_proba output violates the contract."""
    arr = np.asarray(proba)
    if arr.shape != (n_rows, 2):
        return f"predict_proba shape {arr.shape}, expected ({n_rows}, 2)"
    if not np.isfinite(arr).all():
        return "non-finite probabilities"
    if (arr < 0).any() or (arr > 1).any():
        return "probabilities outside [0, 1]"
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
        return "probability rows do not sum to 1"
    return None


def validate_artifact(
    artifact_dir: Path,
    serving_schema: FeatureSchema,
    *,
    expected_version: str | None = None,
    limits: ResourceLimits | None = None,
    smoke: pd.DataFrame | None = None,
) -> ValidationReport:
    limits = limits or ResourceLimits()
    rep = ValidationReport(version=expected_version or artifact_dir.name)

    def add(name: str, ok: bool, detail: str = "") -> bool:
        rep.checks.append(Check(name, ok, detail))
        return ok

    # 1. integrity
    try:
        manifest = verify_integrity(artifact_dir)
    except ArtifactError as e:
        add("integrity", False, e.message if e.code == "integrity" else f"{e.code}: {e.message}")
        return rep
    add("integrity", True, f"sha256 {manifest.model_sha256[:12]}")
    rep.manifest = manifest

    # 2. metadata consistency
    problems = []
    if expected_version and manifest.model_version != expected_version:
        problems.append(f"manifest version {manifest.model_version} != {expected_version}")
    if manifest.model_version != artifact_dir.name:
        problems.append(f"manifest version {manifest.model_version} != dir {artifact_dir.name}")
    if manifest.classes != [0, 1]:
        problems.append(f"classes {manifest.classes} != [0, 1]")
    try:
        declared = FeatureSchema.from_dict(manifest.feature_schema)
        if declared.fingerprint != manifest.feature_schema_fingerprint:
            problems.append(
                f"feature_schema ({declared.version}) does not match its fingerprint "
                f"{manifest.feature_schema_fingerprint[:12]}"
            )
    except (KeyError, TypeError, AssertionError) as e:
        problems.append(f"feature_schema unparseable: {e!r}")
    if not add("metadata", not problems, "; ".join(problems)):
        return rep

    # 3. schema compatibility (declared)
    if manifest.feature_schema_fingerprint != serving_schema.fingerprint:
        add(
            "schema_compatibility",
            False,
            f"artifact schema {manifest.feature_schema.get('version')} "
            f"({manifest.feature_schema_fingerprint[:12]}) != serving "
            f"{serving_schema.version} ({serving_schema.fingerprint[:12]})",
        )
        return rep
    add("schema_compatibility", True, serving_schema.version)

    # 4. load
    t0 = time.perf_counter()
    try:
        est = load_estimator(artifact_dir, manifest)
    except ArtifactError as e:
        add("load", False, f"{e.code}: {e.message}")
        return rep
    load_s = time.perf_counter() - t0
    add("load", True, f"{load_s * 1000:.1f} ms")

    # 5. schema compatibility (actual, catches a manifest that lies)
    names = list(getattr(est, "feature_names_in_", []))
    if not add(
        "model_features",
        names == serving_schema.names,
        "" if names == serving_schema.names else f"model expects {names}",
    ):
        return rep

    # 6. prediction contract + numerical validity on smoke rows
    X = transform(smoke if smoke is not None else smoke_frame(), serving_schema)
    t1 = time.perf_counter()
    try:
        proba = est.predict_proba(X)
    except Exception as e:
        add("prediction_contract", False, f"predict_proba raised {type(e).__name__}: {e}"[:300])
        return rep
    smoke_ms = (time.perf_counter() - t1) * 1000 / max(len(X), 1)
    classes = [int(c) for c in getattr(est, "classes_", [])]
    if not add("prediction_contract", classes == [0, 1], f"classes_={classes}"):
        return rep
    err = check_outputs(proba, len(X))
    if not add("numerical_validity", err is None, err or f"{len(X)} smoke rows"):
        return rep

    # 7. resources (measurable on this host only)
    ok = load_s <= limits.max_load_seconds and smoke_ms <= limits.max_smoke_ms_per_row
    add(
        "resources",
        ok,
        f"load {load_s:.2f}s (<= {limits.max_load_seconds}), "
        f"{smoke_ms:.2f} ms/row (<= {limits.max_smoke_ms_per_row})",
    )
    if rep.passed:
        rep.estimator = est
    return rep
