"""Model artifact packaging, integrity verification and safe loading.

Artifact layout (one directory per model version)::

    <version>/
      model.skops      # skops-serialized estimator (no arbitrary code execution on load)
      manifest.json    # metadata, provenance, schema, sha256 of model.skops

Security model
--------------
* New artifacts use ``skops.io``, never pickle. Loading goes through
  ``skops.io.get_untrusted_types`` and only an explicit allowlist of extra types is
  trusted. Any other type is a rejection, not a prompt.
* Integrity is a sha256 over the model file, recorded in the manifest. This detects
  corruption and accidental modification. It is NOT a signature: anyone who can
  write both files can forge it. Authenticity requires signing (not implemented).
* The single legacy pickle (``import_legacy_pickle``) is unpickled only after its
  digest matches a pin; pickle loading is otherwise never used.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn
import skops
import skops.io as sio
from pydantic import BaseModel, ConfigDict, Field

from mlpipe.schema import FeatureSchema

MODEL_FILE = "model.skops"
MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT = "mlpipe-artifact/1"
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Extra (non-default-trusted) types an artifact may contain. Reviewed by hand.
TRUSTED_EXTRA_TYPES: frozenset[str] = frozenset({"sklearn.tree._tree.Tree"})


class ArtifactError(Exception):
    """Raised for any artifact that must not be served. ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    format: str = MANIFEST_FORMAT
    model_version: str
    created_at: str
    model_file: str = MODEL_FILE
    model_sha256: str
    model_bytes: int
    estimator: str
    classes: list[int]
    decision_threshold: float = Field(ge=0.0, le=1.0)
    feature_schema: dict[str, Any]
    feature_schema_fingerprint: str
    framework: dict[str, str]
    training: dict[str, Any]
    metrics: dict[str, float]
    reference_stats: dict[str, dict[str, float]]
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def framework_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "skops": skops.__version__,
    }


def write_json_atomic(path: Path, obj: Any) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def package_model(
    estimator: Any,
    out_dir: Path,
    *,
    model_version: str,
    schema: FeatureSchema,
    training: dict[str, Any],
    metrics: dict[str, float],
    reference_stats: dict[str, dict[str, float]],
    decision_threshold: float = 0.5,
    provenance: dict[str, Any] | None = None,
) -> Manifest:
    if not VERSION_RE.match(model_version):
        raise ArtifactError("invalid_version", f"bad model version {model_version!r}")
    if out_dir.exists():
        raise ArtifactError("exists", f"{out_dir} already exists")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{model_version}."))
    try:
        model_path = tmp / MODEL_FILE
        sio.dump(estimator, model_path)
        manifest = Manifest(
            model_version=model_version,
            created_at=datetime.now(UTC).isoformat(),
            model_sha256=sha256_file(model_path),
            model_bytes=model_path.stat().st_size,
            estimator=f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            classes=[int(c) for c in getattr(estimator, "classes_", [0, 1])],
            decision_threshold=decision_threshold,
            feature_schema=schema.to_dict(),
            feature_schema_fingerprint=schema.fingerprint,
            framework=framework_versions(),
            training=training,
            metrics=metrics,
            reference_stats=reference_stats,
            provenance={"git_commit": git_commit(), **(provenance or {})},
        )
        write_json_atomic(tmp / MANIFEST_FILE, manifest.model_dump())
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def read_manifest(artifact_dir: Path) -> Manifest:
    path = artifact_dir / MANIFEST_FILE
    if not path.is_file():
        raise ArtifactError("manifest_missing", f"no manifest in {artifact_dir}")
    try:
        return Manifest.model_validate_json(path.read_text())
    except ValueError as e:
        raise ArtifactError("manifest_invalid", str(e).splitlines()[0]) from e


def verify_integrity(artifact_dir: Path) -> Manifest:
    """Manifest parses, model file is where it should be, size bounded, hash matches."""
    manifest = read_manifest(artifact_dir)
    if manifest.format != MANIFEST_FORMAT:
        raise ArtifactError("manifest_format", f"unsupported format {manifest.format}")
    if manifest.model_file != MODEL_FILE:
        raise ArtifactError("manifest_invalid", "model_file must be model.skops")
    model_path = artifact_dir / MODEL_FILE
    if not model_path.is_file() or model_path.is_symlink():
        raise ArtifactError("model_missing", "model file missing or a symlink")
    size = model_path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise ArtifactError("too_large", f"{size} bytes > {MAX_ARTIFACT_BYTES}")
    if size != manifest.model_bytes:
        raise ArtifactError("integrity", f"size {size} != manifest {manifest.model_bytes}")
    digest = sha256_file(model_path)
    if digest != manifest.model_sha256:
        raise ArtifactError("integrity", f"sha256 {digest[:12]} != manifest")
    return manifest


def load_estimator(artifact_dir: Path, manifest: Manifest) -> Any:
    """Load with skops, trusting only the default safe set plus the reviewed allowlist.

    The caller must have run ``verify_integrity`` first; the hash is rechecked here
    on the exact bytes that are deserialized, closing the check/use gap.
    """
    data = (artifact_dir / MODEL_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.model_sha256:
        raise ArtifactError("integrity", "model bytes changed between verify and load")
    try:
        untrusted = set(sio.get_untrusted_types(data=data))
    except Exception as e:
        raise ArtifactError("load_failed", f"unreadable skops archive: {type(e).__name__}") from e
    unexpected = untrusted - TRUSTED_EXTRA_TYPES
    if unexpected:
        raise ArtifactError("untrusted_types", f"refusing types {sorted(unexpected)}")
    try:
        return sio.loads(data, trusted=sorted(untrusted))
    except Exception as e:
        raise ArtifactError("load_failed", f"{type(e).__name__}: {e}"[:300]) from e


# --- legacy migration --------------------------------------------------------------

# md5 of models/model.pkl as pinned in dvc.lock at commit 37995fe, and the sha256 of
# the same bytes (preserved in mlruns/0/models/m-ac8443.../artifacts/model.pkl).
LEGACY_PICKLE_MD5 = "333ad164dacb32f0d8844a27d7876c67"
LEGACY_PICKLE_SHA256 = "57f513c612dd6398552cfb78a103bdab069d54115a4e1c2ab7684850535febda"
LEGACY_PICKLE_PATH = Path("mlruns/0/models/m-ac8443cec0b64c03929f2d97d344b3c5/artifacts/model.pkl")


def load_legacy_pickle(path: Path) -> Any:
    """Unpickle the one legacy artifact, only if it is byte-identical to the DVC pin."""
    data = path.read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    sha = hashlib.sha256(data).hexdigest()
    if md5 != LEGACY_PICKLE_MD5 or sha != LEGACY_PICKLE_SHA256:
        raise ArtifactError("legacy_pin", f"md5 {md5} / sha256 {sha[:12]} do not match pins")
    return pickle.loads(data)
