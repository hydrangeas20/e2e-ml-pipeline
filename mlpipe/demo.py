"""End-to-end rollout demonstration against a real uvicorn server process.

Every step asserts on observed HTTP responses and persisted registry state; the
script exits non-zero if any expectation fails. Nothing is printed as a success
unless it was observed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from mlpipe.fixtures import FIXTURES, build_fixture
from mlpipe.registry import Registry
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.deployment import DeploymentController
from mlpipe.serving.observability import Metrics
from mlpipe.train import import_legacy, train_and_package
from mlpipe.validation import smoke_frame

TOKEN = "demo-admin-token-0123456789"
GATE = {"MLPIPE_GATE_MIN_SAMPLES": "10", "MLPIPE_GATE_WINDOW": "200",
        "MLPIPE_GATE_MAX_ERROR_RATE": "0.2", "MLPIPE_GATE_MAX_CONSECUTIVE_FAILURES": "5"}


class DemoFailure(AssertionError):
    pass


def expect(cond: bool, what: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + what, flush=True)
    if not cond:
        raise DemoFailure(what)


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def run_demo(workdir: Path, *, port: int, data: Path) -> int:
    if workdir.exists():
        shutil.rmtree(workdir)
    reg_dir = workdir / "registry"
    reg = Registry(reg_dir)
    report: dict[str, Any] = {"steps": []}
    proc: subprocess.Popen[bytes] | None = None
    log_path = workdir / "server.log"
    try:
        step("1. prepare artifacts (legacy import, new candidate, bad fixtures)")
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            import_legacy(t / "v1-legacy-rf")
            reg.register(t / "v1-legacy-rf")
            train_and_package(data, t / "v2-rf", version="v2-rf")
            reg.register(t / "v2-rf")
            for kind in FIXTURES:
                reg.register(build_fixture(kind, data, t))
        ctl = DeploymentController(reg, SERVING_SCHEMA, _settings(reg_dir), Metrics())
        boot = ctl.bootstrap("v1-legacy-rf")
        expect(boot["result"] == "active", "bootstrap v1-legacy-rf as active + known-good")

        step("2. start the service (uvicorn subprocess)")
        env = {**os.environ, "MLPIPE_REGISTRY_DIR": str(reg_dir),
               "MLPIPE_AUDIT_LOG_PATH": str(workdir / "predictions.jsonl"),
               "MLPIPE_ENABLE_FAULT_INJECTION": "true", "MLPIPE_ADMIN_TOKEN": TOKEN, **GATE}
        log_f = log_path.open("wb")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port),
             "--host", "127.0.0.1", "--timeout-graceful-shutdown", "10"],
            env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        c = httpx.Client(base_url=base, timeout=10,
                         headers={"Authorization": f"Bearer {TOKEN}"})
        _wait_ready(c)
        expect(c.get("/readyz").status_code == 200, "/readyz is 200")

        rows = smoke_frame().head(2).to_dict(orient="records")

        def predict() -> httpx.Response:
            return c.post("/v1/predict", json={"instances": rows})

        step("3. inference with the known-good model")
        r = predict()
        expect(r.status_code == 200 and r.json()["model_version"] == "v1-legacy-rf",
               f"predict -> 200 from v1-legacy-rf ({r.json().get('predictions')})")

        step("4. pre-promotion rejection of invalid candidates")
        for kind in FIXTURES:
            v = f"bad-{kind.replace('_', '-')}"
            r = c.post("/admin/promote", json={"version": v})
            body = r.json()
            reason = (body.get("validation") or {}).get("failure_reason") or body
            expect(r.status_code == 422 and body.get("result") == "rejected",
                   f"{v} rejected: {reason}")
            ok = predict()
            expect(ok.status_code == 200 and ok.json()["model_version"] == "v1-legacy-rf",
                   "  active model undisturbed, inference still 200 from v1-legacy-rf")
            report["steps"].append({"rejected": v, "reason": reason})

        step("5. promote a valid candidate (atomic reference swap, probation starts)")
        gen = c.get("/admin/deployment").json()["registry"]["generation"]
        stale = c.post("/admin/promote", json={"version": "v2-rf", "expected_generation": gen - 1})
        expect(stale.status_code == 409 and stale.json()["error"]["code"] == "stale_generation",
               "promotion with a stale generation is refused (409)")
        r = c.post("/admin/promote", json={"version": "v2-rf", "expected_generation": gen})
        expect(r.status_code == 200 and r.json()["result"] == "active_on_probation",
               f"v2-rf promoted: {r.json()}")
        served = [predict().json()["model_version"] for _ in range(5)]
        expect(served == ["v2-rf"] * 5, "5 predictions served by v2-rf")

        step("6. inject a post-promotion failure: v2-rf errors after 3 more requests")
        r = c.post("/admin/faults", json={"kind": "error_after_n", "model_version": "v2-rf",
                                          "n": 3})
        expect(r.status_code == 200, f"fault armed: {r.json()}")
        trace = []
        for _ in range(30):
            resp = predict()
            j = resp.json()
            mv = j.get("model_version") or j.get("error", {}).get("details", {}).get(
                "model_version")
            trace.append((resp.status_code, mv))
            if resp.status_code == 200 and mv == "v1-legacy-rf":
                break
        print("  observed (status, model_version):", trace)
        report["fault_trace"] = trace
        n_err = sum(1 for s, _ in trace if s >= 500)
        expect(1 <= n_err <= int(GATE["MLPIPE_GATE_MAX_CONSECUTIVE_FAILURES"]),
               f"{n_err} failed requests before detection (bounded by the gate)")

        step("7. detection and automatic restore of the known-good model")
        dep = c.get("/admin/deployment").json()
        expect(dep["active"] == "v1-legacy-rf" and dep["probation"] is None,
               f"active={dep['active']}, probation cleared")
        states = dep["registry"]["versions"]
        expect(states["v2-rf"]["state"] == "rolled_back", "v2-rf persisted as rolled_back")
        expect(dep["registry"]["active"] == "v1-legacy-rf", "registry state.json active=v1")

        step("8. inference succeeds after recovery")
        r = predict()
        expect(r.status_code == 200 and r.json()["model_version"] == "v1-legacy-rf",
               "predict -> 200 from v1-legacy-rf")
        r = c.post("/admin/rollback", json={"reason": "demo idempotency check"})
        expect(r.json()["result"] == "noop", "second (manual) rollback is an idempotent no-op")
        r = c.post("/admin/promote", json={"version": "v2-rf"})
        expect(r.status_code == 409, "rolled-back version cannot be re-promoted (409)")

        step("9. deployment events and metrics")
        events = c.get("/admin/events", params={"limit": 200}).json()
        for e in events:
            print(f"  gen={e.get('generation'):>3} {e['action']:<18} {e.get('model_version')!s:<20}"
                  f" {e.get('from_state')!s:>10} -> {e.get('to_state')!s:<12}"
                  f" {e.get('rollback_trigger') or ''} {e.get('failure_reason') or ''}"[:160])
        rb = [e for e in events if e["action"] == "rollback"]
        expect(len(rb) == 1 and rb[0]["rollback_trigger"] == "health_gate"
               and rb[0]["recovery_result"] == "verified",
               "exactly one rollback event: trigger=health_gate, recovery_result=verified")
        prom = c.get("/metrics").text
        keep = [ln for ln in prom.splitlines() if ln.startswith((
            "mlpipe_rollbacks_total", "mlpipe_active_model_info", "mlpipe_prediction_errors",
            "mlpipe_deployment_events_total", "mlpipe_model_load_failures", "mlpipe_predictions"))]
        print("\n".join("  " + k for k in keep))
        report["metrics"] = keep
        report["events"] = events

        step("10. graceful shutdown")
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=20)
        proc = None
        log_f.close()
        server_log = log_path.read_text()
        # uvicorn >= 0.30 re-raises the captured signal after a clean shutdown, so a
        # graceful exit shows rc -15; the lifespan "service stopped" line is the proof.
        expect(rc in (0, -signal.SIGTERM) and '"msg": "service stopped"' in server_log
               and "Application shutdown complete" in server_log,
               f"server drained and shut down via lifespan on SIGTERM (rc={rc})")
        audit_lines = (workdir / "predictions.jsonl").read_text().splitlines()
        versions = {json.loads(x)["model_version"] for x in audit_lines}
        expect(versions == {"v1-legacy-rf", "v2-rf"},
               f"{len(audit_lines)} audit records, each attributed to a model version")
        report["result"] = "passed"
        print("\nDEMO PASSED")
        return 0
    except DemoFailure:
        report["result"] = "failed"
        print(f"\nDEMO FAILED (server log: {log_path})")
        return 1
    finally:
        if proc is not None:
            proc.kill()
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "demo_report.json").write_text(json.dumps(report, indent=2, default=str))


def _settings(reg_dir: Path) -> Any:
    from mlpipe.config import Settings

    return Settings(registry_dir=reg_dir)


def _wait_ready(c: httpx.Client, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if c.get("/readyz").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise DemoFailure("server did not become ready")
