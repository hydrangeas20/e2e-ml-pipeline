"""Layered model-quality diagnosis.

Walks the layers in pipeline order and reports the first one that is unhealthy as
the most likely root cause, because every downstream layer inherits upstream
problems (a broken transform also lowers AUC, but the AUC drop is not the cause):

  1. data quality      raw input: schema/bounds, missing-as-zero rates vs training
  2. feature pipeline  served features vs the same rows through the reference transform
  3. model quality     ranking quality (ROC-AUC) vs the artifact's recorded metric  [labels]
  4. calibration       Brier/ECE and positive-rate vs label prevalence              [labels]
  5. serving           error/timeout/rejection counters from /metrics (optional)

Without labels (the normal case when outcomes arrive late), layers 3-4 are reported
as ``unknown``: proxy signals (input shift) are never presented as model quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from mlpipe.artifacts import load_estimator, verify_integrity
from mlpipe.features import standardized_mean_shift, transform
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA, TARGET

SHIFT_ALERT = 0.5  # standardized mean shift
ZERO_RATE_ALERT = 0.15  # absolute increase in zero (missing) fraction
AUC_DROP_ALERT = 0.05
ECE_ALERT = 0.10
POS_RATE_ALERT = 0.15


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def diagnose(
    est: Any,
    manifest: dict[str, Any],
    raw: pd.DataFrame,
    *,
    defect: str | None = None,
    labels: bool = True,
    server_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    ref = manifest["reference_stats"]
    layers: dict[str, dict[str, Any]] = {}

    # 1. data quality
    issues = []
    for spec in SERVING_SCHEMA.features:
        if spec.name not in raw.columns:
            issues.append(f"{spec.name}: missing column")
            continue
        col = pd.to_numeric(raw[spec.name], errors="coerce")
        if col.isna().any():
            issues.append(f"{spec.name}: {int(col.isna().sum())} non-numeric")
        oob = ((col < spec.minimum) | (col > spec.maximum)).sum()
        if oob:
            issues.append(f"{spec.name}: {int(oob)} out of contract bounds")
        if spec.name in ref and spec.name not in ("Pregnancies",):
            dz = float((col == 0).mean()) - float(ref[spec.name]["zero_fraction"])
            if dz > ZERO_RATE_ALERT:
                issues.append(f"{spec.name}: zero/missing rate +{dz:.2f} vs training")
    layers["data_quality"] = {"status": "problem" if issues else "ok", "issues": issues}
    if any("missing column" in i or "non-numeric" in i for i in issues):
        return _verdict(layers)

    # 2. feature pipeline: compare served transform against the reference transform
    X_ref = transform(raw, SERVING_SCHEMA)
    X_srv = transform(raw, SERVING_SCHEMA, defect=defect)
    diff = (X_srv - X_ref).abs().max()
    changed = {k: float(v) for k, v in diff.items() if v > 1e-9}
    shift_srv = standardized_mean_shift(X_srv, ref)
    shift_raw = standardized_mean_shift(X_ref, ref)
    alerts = {
        k: round(v, 3)
        for k, v in shift_srv.items()
        if v > SHIFT_ALERT and shift_raw.get(k, 0) <= SHIFT_ALERT
    }
    layers["feature_pipeline"] = {
        "status": "problem" if changed or alerts else "ok",
        "features_differing_from_reference_transform": sorted(changed),
        "served_shift_alerts": alerts,
        "raw_input_shift": {k: round(v, 3) for k, v in shift_raw.items() if v > SHIFT_ALERT},
    }

    # 3-4. need labels
    if not labels or TARGET not in raw.columns:
        layers["model_quality"] = {"status": "unknown", "why": "labels unavailable (delayed)"}
        layers["calibration"] = {"status": "unknown", "why": "labels unavailable (delayed)"}
    else:
        y = raw[TARGET].to_numpy()
        p_srv = est.predict_proba(X_srv)[:, 1]
        p_ref = est.predict_proba(X_ref)[:, 1]
        auc_srv, auc_ref = float(roc_auc_score(y, p_srv)), float(roc_auc_score(y, p_ref))
        recorded = float(manifest.get("metrics", {}).get("roc_auc", auc_ref))
        layers["model_quality"] = {
            "status": "problem" if recorded - auc_ref > AUC_DROP_ALERT else "ok",
            "roc_auc_reference_transform": round(auc_ref, 4),
            "roc_auc_served": round(auc_srv, 4),
            "roc_auc_recorded_at_training": round(recorded, 4),
            "n": len(y),
        }
        thr = float(manifest.get("decision_threshold", 0.5))
        pos_rate = float((p_ref >= thr).mean())
        e = ece(y, p_ref)
        cal_issues = []
        if e > ECE_ALERT:
            cal_issues.append(f"ECE {e:.3f} > {ECE_ALERT}")
        if abs(pos_rate - float(y.mean())) > POS_RATE_ALERT:
            cal_issues.append(f"positive rate {pos_rate:.2f} vs prevalence {y.mean():.2f}")
        layers["calibration"] = {
            "status": "problem" if cal_issues else "ok",
            "issues": cal_issues,
            "ece": round(e, 4),
            "brier": round(float(brier_score_loss(y, p_ref)), 4),
            "threshold": thr,
        }

    # 5. serving
    if server_metrics is not None:
        bad = {k: v for k, v in server_metrics.items() if v > 0}
        layers["serving"] = {"status": "problem" if bad else "ok", "counters": bad}
    else:
        layers["serving"] = {"status": "unknown", "why": "no server metrics supplied"}
    return _verdict(layers)


def _verdict(layers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    order = ["data_quality", "feature_pipeline", "model_quality", "calibration", "serving"]
    first = next((name for name in order if layers.get(name, {}).get("status") == "problem"), None)
    return {"likely_root_cause": first or "none_detected", "layers": layers}


def diagnose_file(
    registry_dir: Path, version: str, data: Path, *, defect: str | None = None, labels: bool = True
) -> dict[str, Any]:
    reg = Registry(registry_dir)
    d = reg.artifact_dir(version)
    manifest = verify_integrity(d)
    est = load_estimator(d, manifest)
    rep = diagnose(est, manifest.model_dump(), pd.read_csv(data), defect=defect, labels=labels)
    return {"model_version": version, "data": str(data), "simulated_defect": defect, **rep}
