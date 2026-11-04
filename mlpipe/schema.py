"""Versioned feature schema shared by training, validation and serving.

The schema is the serving contract. A model artifact declares the schema it was
trained against; the server refuses any artifact whose schema fingerprint differs
from the one it serves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

TARGET = "Outcome"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: Literal["int", "float"]
    minimum: float
    maximum: float


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    features: tuple[FeatureSpec, ...]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "features": [asdict(f) for f in self.features]}

    @property
    def fingerprint(self) -> str:
        canon = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> FeatureSchema:
        feats = d["features"]
        assert isinstance(feats, list)
        return cls(
            version=str(d["version"]),
            features=tuple(FeatureSpec(**f) for f in feats),
        )


# Bounds are contract limits (reject physically impossible / garbage input), not
# clinical plausibility limits. Zeros are legal: the source dataset encodes missing
# measurements as 0, and the diagnostics module flags them as a data-quality issue.
GDM_SCHEMA_V1 = FeatureSchema(
    version="gdm-features/1",
    features=(
        FeatureSpec("Pregnancies", "int", 0, 30),
        FeatureSpec("Glucose", "int", 0, 400),
        FeatureSpec("BloodPressure", "int", 0, 250),
        FeatureSpec("SkinThickness", "int", 0, 120),
        FeatureSpec("Insulin", "int", 0, 1000),
        FeatureSpec("BMI", "float", 0, 90),
        FeatureSpec("DiabetesPedigreeFunction", "float", 0, 3),
        FeatureSpec("Age", "int", 0, 120),
    ),
)

SERVING_SCHEMA = GDM_SCHEMA_V1
