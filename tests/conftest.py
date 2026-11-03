from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipe.config import Settings
from mlpipe.fixtures import FIXTURES, build_fixture
from mlpipe.registry import Registry
from mlpipe.serving.app import create_app
from mlpipe.train import import_legacy, train_and_package
from mlpipe.validation import smoke_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "splits" / "test.csv"
TOKEN = "test-admin-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Built once per session: legacy import, two good candidates, all bad fixtures."""
    root = tmp_path_factory.mktemp("artifacts")
    out = {"v1-legacy-rf": root / "v1-legacy-rf"}
    import_legacy(out["v1-legacy-rf"], pickle_path=ROOT / "mlruns/0/models/"
                  "m-ac8443cec0b64c03929f2d97d344b3c5/artifacts/model.pkl",
                  holdout_csv=DATA)
    for v, p in (("v2-rf", {}), ("v3-rf", {"n_estimators": 50, "random_state": 7})):
        out[v] = root / v
        train_and_package(DATA, out[v], version=v, params=p)
    for kind in FIXTURES:
        d = build_fixture(kind, DATA, root)
        out[d.name] = d
    return out


@pytest.fixture
def registry(tmp_path: Path, artifacts: dict[str, Path]) -> Registry:
    reg = Registry(tmp_path / "registry")
    for d in artifacts.values():
        reg.register(d)
    return reg


@pytest.fixture
def settings_factory(tmp_path: Path, registry: Registry) -> Callable[..., Settings]:
    def make(**kw: Any) -> Settings:
        base: dict[str, Any] = dict(
            registry_dir=registry.root,
            audit_log_path=tmp_path / "predictions.jsonl",
            enable_fault_injection=True,
            admin_token=TOKEN,
            gate_min_samples=10,
            gate_window=30,
            gate_max_error_rate=0.2,
            gate_max_consecutive_failures=5,
        )
        base.update(kw)
        return Settings(**base)

    return make


@pytest.fixture
def bootstrapped(registry: Registry, settings_factory: Callable[..., Settings]) -> Registry:
    from mlpipe.schema import SERVING_SCHEMA
    from mlpipe.serving.deployment import DeploymentController
    from mlpipe.serving.observability import Metrics

    DeploymentController(registry, SERVING_SCHEMA, settings_factory(), Metrics()).bootstrap(
        "v1-legacy-rf"
    )
    return registry


@pytest.fixture
def client_factory(
    settings_factory: Callable[..., Settings],
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(**kw: Any) -> TestClient:
        c = TestClient(create_app(settings_factory(**kw)), headers=AUTH)
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(bootstrapped: Registry, client_factory: Callable[..., TestClient]) -> TestClient:
    return client_factory()


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return smoke_frame().head(3).to_dict(orient="records")  # type: ignore[no-any-return]


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)
