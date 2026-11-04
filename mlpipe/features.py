"""Feature transformation shared by training and serving.

The production transform is deliberately simple: column selection, ordering and
float64 casting against the versioned schema. Keeping one function for both paths
removes training/serving skew by construction.

``DEFECTS`` holds deliberately broken transforms used by fault injection and the
diagnostics workflow. They are never selected unless fault injection is enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from mlpipe.schema import FeatureSchema


class FeatureValidationError(ValueError):
    pass


def _glucose_unit_bug(df: pd.DataFrame) -> pd.DataFrame:
    # Simulates a unit-conversion regression: mg/dL silently converted to mmol/L.
    out = df.copy()
    out["Glucose"] = out["Glucose"] / 18.0
    return out


def _column_swap_bug(df: pd.DataFrame) -> pd.DataFrame:
    # Simulates a column-order regression in an upstream join.
    out = df.copy()
    out[["BMI", "Age"]] = out[["Age", "BMI"]].to_numpy()
    return out


DEFECTS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "glucose_mmol": _glucose_unit_bug,
    "bmi_age_swap": _column_swap_bug,
}


def transform(
    raw: pd.DataFrame, schema: FeatureSchema, *, defect: str | None = None
) -> pd.DataFrame:
    missing = [c for c in schema.names if c not in raw.columns]
    if missing:
        raise FeatureValidationError(f"missing features: {missing}")
    df = raw[schema.names].astype("float64")
    if defect is not None:
        df = DEFECTS[defect](df)
    values = df.to_numpy()
    if not np.isfinite(values).all():
        raise FeatureValidationError("non-finite feature values after transform")
    return df


def reference_stats(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-feature training distribution summary stored in the artifact manifest."""
    stats: dict[str, dict[str, float]] = {}
    for col in features.columns:
        s = features[col].astype("float64")
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "zero_fraction": float((s == 0).mean()),
        }
    return stats


def standardized_mean_shift(
    features: pd.DataFrame, ref: dict[str, dict[str, Any]]
) -> dict[str, float]:
    """|mean_live - mean_ref| / std_ref per feature (a proxy drift signal, no labels)."""
    out: dict[str, float] = {}
    for col in features.columns:
        r = ref.get(col)
        if not r:
            continue
        std = float(r["std"]) or 1.0
        out[col] = abs(float(features[col].mean()) - float(r["mean"])) / std
    return out
