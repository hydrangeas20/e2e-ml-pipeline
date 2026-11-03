"""``mlpipe`` command-line interface.

Offline commands operate on the registry directory directly (fcntl-locked):
  fetch-data, import-legacy, train, make-fixtures, register, validate, bootstrap,
  state, events, diagnose
Online commands talk to a running server (``--url``, admin token from
``MLPIPE_ADMIN_TOKEN``) because the server owns the in-memory active model:
  predict, promote, rollback, inject-fault, clear-faults, server-state
``demo`` runs the full rollout / failure / automatic-rollback scenario end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from mlpipe.config import Settings
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _controller(registry_dir: Path) -> Any:
    from mlpipe.serving.deployment import DeploymentController
    from mlpipe.serving.observability import Metrics

    return DeploymentController(
        Registry(registry_dir), SERVING_SCHEMA, Settings(registry_dir=registry_dir), Metrics()
    )


def _client(args: argparse.Namespace) -> httpx.Client:
    token = os.environ.get("MLPIPE_ADMIN_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=args.url, headers=headers, timeout=30)


def _show(r: httpx.Response) -> int:
    try:
        _print(r.json())
    except ValueError:
        print(r.text)
    return 0 if r.status_code < 400 else 1


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from mlpipe.train import PUBLIC_DATA_SHA256, PUBLIC_DATA_URL

    with urllib.request.urlopen(PUBLIC_DATA_URL, timeout=30) as resp:  # noqa: S310
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PUBLIC_DATA_SHA256:
        print(f"sha256 mismatch: {digest}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"wrote {out} sha256={digest}")
    return 0


def _register_from(build: Any, registry: Registry, version: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / version
        build(d)
        return registry.register(d)


def cmd_import_legacy(args: argparse.Namespace) -> int:
    from mlpipe.train import import_legacy

    reg = Registry(Path(args.registry))
    v = _register_from(lambda d: import_legacy(d, version=args.version), reg, args.version)
    _print({"registered": v, "state": str(reg.state_of(v))})
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from mlpipe.train import train_and_package

    reg = Registry(Path(args.registry))
    params = json.loads(args.params) if args.params else None
    v = _register_from(
        lambda d: train_and_package(Path(args.data), d, version=args.version, params=params),
        reg, args.version,
    )
    m = json.loads((reg.artifact_dir(v) / "manifest.json").read_text())
    _print({"registered": v, "metrics": m["metrics"], "sha256": m["model_sha256"]})
    return 0


def cmd_make_fixtures(args: argparse.Namespace) -> int:
    from mlpipe.fixtures import FIXTURES, build_fixture

    reg = Registry(Path(args.registry))
    out = []
    for kind in FIXTURES:
        with tempfile.TemporaryDirectory() as tmp:
            d = build_fixture(kind, Path(args.data), Path(tmp), prefix=args.prefix)
            out.append(reg.register(d))
    _print({"registered": out})
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    reg = Registry(Path(args.registry))
    _print({"registered": reg.register(Path(args.artifact_dir))})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    if args.path:
        from mlpipe.validation import validate_artifact

        res = validate_artifact(Path(args.version), SERVING_SCHEMA).summary()
    else:
        res = _controller(Path(args.registry)).validate(args.version)
    _print(res)
    return 0 if res["passed"] else 2


def cmd_bootstrap(args: argparse.Namespace) -> int:
    _print(_controller(Path(args.registry)).bootstrap(args.version))
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    _print(Registry(Path(args.registry)).inspect())
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    for e in Registry(Path(args.registry)).events(limit=args.limit):
        print(json.dumps(e, sort_keys=True))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.json).read_text()) if args.json else None
    if rows is None:
        from mlpipe.validation import smoke_frame

        rows = smoke_frame().head(args.n).to_dict(orient="records")
    with _client(args) as c:
        return _show(c.post("/v1/predict", json={"instances": rows}))


def cmd_promote(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {"version": args.version}
    if args.expected_generation is not None:
        body["expected_generation"] = args.expected_generation
    with _client(args) as c:
        return _show(c.post("/admin/promote", json=body))


def cmd_rollback(args: argparse.Namespace) -> int:
    with _client(args) as c:
        return _show(c.post("/admin/rollback", json={"reason": args.reason}))


def cmd_inject(args: argparse.Namespace) -> int:
    body = {k: v for k, v in {"kind": args.kind, "model_version": args.model_version,
                              "n": args.n, "defect": args.defect,
                              "seconds": args.seconds}.items() if v is not None}
    with _client(args) as c:
        return _show(c.post("/admin/faults", json=body))


def cmd_clear(args: argparse.Namespace) -> int:
    with _client(args) as c:
        return _show(c.delete("/admin/faults"))


def cmd_server_state(args: argparse.Namespace) -> int:
    with _client(args) as c:
        return _show(c.get("/admin/deployment"))


def cmd_diagnose(args: argparse.Namespace) -> int:
    from mlpipe.diagnose import diagnose_file

    rep = diagnose_file(Path(args.registry), args.version, Path(args.data),
                        defect=args.simulate_defect, labels=not args.no_labels)
    _print(rep)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from mlpipe.demo import run_demo

    return run_demo(Path(args.workdir), port=args.port, data=Path(args.data))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlpipe")
    sub = p.add_subparsers(dest="cmd", required=True)
    reg_default = os.environ.get("MLPIPE_REGISTRY_DIR", "registry")
    url_default = os.environ.get("MLPIPE_URL", "http://127.0.0.1:8000")

    def add(name: str, fn: Any, *, registry: bool = False, url: bool = False) -> Any:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        if registry:
            sp.add_argument("--registry", default=reg_default)
        if url:
            sp.add_argument("--url", default=url_default)
        return sp

    add("fetch-data", cmd_fetch_data).add_argument("--out", default="data/public/diabetes.csv")
    add("import-legacy", cmd_import_legacy, registry=True).add_argument(
        "--version", default="v1-legacy-rf")
    sp = add("train", cmd_train, registry=True)
    sp.add_argument("--data", default="data/public/diabetes.csv")
    sp.add_argument("--version", required=True)
    sp.add_argument("--params", help="JSON dict overriding DEFAULT_PARAMS")
    sp = add("make-fixtures", cmd_make_fixtures, registry=True)
    sp.add_argument("--data", default="data/splits/test.csv")
    sp.add_argument("--prefix", default="bad")
    add("register", cmd_register, registry=True).add_argument("artifact_dir")
    sp = add("validate", cmd_validate, registry=True)
    sp.add_argument("version", help="registered version, or a directory with --path")
    sp.add_argument("--path", action="store_true")
    add("bootstrap", cmd_bootstrap, registry=True).add_argument("version")
    add("state", cmd_state, registry=True)
    add("events", cmd_events, registry=True).add_argument("--limit", type=int, default=50)
    sp = add("predict", cmd_predict, url=True)
    sp.add_argument("--json", help="file with a JSON list of feature dicts")
    sp.add_argument("-n", type=int, default=1)
    sp = add("promote", cmd_promote, url=True)
    sp.add_argument("version")
    sp.add_argument("--expected-generation", type=int)
    add("rollback", cmd_rollback, url=True).add_argument("--reason", default="manual")
    sp = add("inject-fault", cmd_inject, url=True)
    sp.add_argument("kind", choices=["error_after_n", "feature_defect", "audit_failure", "slow"])
    sp.add_argument("--model-version")
    sp.add_argument("--n", type=int)
    sp.add_argument("--defect")
    sp.add_argument("--seconds", type=float)
    add("clear-faults", cmd_clear, url=True)
    add("server-state", cmd_server_state, url=True)
    sp = add("diagnose", cmd_diagnose, registry=True)
    sp.add_argument("--version", required=True)
    sp.add_argument("--data", default="data/splits/test.csv")
    sp.add_argument("--simulate-defect")
    sp.add_argument("--no-labels", action="store_true")
    sp = add("demo", cmd_demo)
    sp.add_argument("--workdir", default="demo_run")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--data", default="data/splits/test.csv")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.fn(args)
    except Exception as e:  # report cleanly; exit non-zero
        code = getattr(e, "code", type(e).__name__)
        print(json.dumps({"error": code, "message": str(e)}), file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
