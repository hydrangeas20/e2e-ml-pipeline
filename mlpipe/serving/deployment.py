"""Deployment controller: validation, atomic promotion, health-gated probation and
automatic rollback to the last known-good model.

Invariants
----------
1. A version becomes ACTIVE only after ``validate_artifact`` passes *at promotion
   time* (re-verified, so an artifact modified after validation is caught).
2. The known-good model stays loaded in memory for the whole probation, so rollback
   is a reference swap that cannot fail on I/O or deserialization.
3. Each request reads the active ``LoadedModel`` reference exactly once and uses it
   for inference, the audit record, the response and the health-gate outcome. A
   request is therefore always attributed to the model that actually served it.
4. Durable state is committed before in-memory state changes on promotion; on
   rollback memory changes first (stop serving the bad model even if disk fails),
   and a restart with an unfinished probation always restores known-good.
5. At most one promotion at a time; a promotion is refused while another version is
   on probation, and when ``expected_generation`` is stale.

Not implemented (and not claimed): canary / weighted traffic splitting, shadow
traffic, multi-process or multi-host coordination.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, NoReturn

import numpy as np
import pandas as pd

from mlpipe.artifacts import Manifest, sha256_file
from mlpipe.config import Settings
from mlpipe.crash import crashpoint
from mlpipe.features import standardized_mean_shift
from mlpipe.registry import Registry, RegistryError, State
from mlpipe.schema import FeatureSchema
from mlpipe.serving.observability import Metrics, log
from mlpipe.validation import ResourceLimits, ValidationReport, validate_artifact

logger = logging.getLogger("mlpipe.deployment")


class DeploymentError(Exception):
    def __init__(self, code: str, message: str, status: int = 409) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True, eq=False)
class LoadedModel:
    version: str
    manifest: Manifest
    estimator: Any
    loaded_at: str

    @property
    def sha256(self) -> str:
        return self.manifest.model_sha256

    @property
    def threshold(self) -> float:
        return self.manifest.decision_threshold


@dataclass
class GateConfig:
    min_samples: int
    window: int
    max_error_rate: float
    max_consecutive_failures: int
    max_p95_latency_ms: float

    @classmethod
    def from_settings(cls, s: Settings) -> GateConfig:
        return cls(
            s.gate_min_samples,
            s.gate_window,
            s.gate_max_error_rate,
            s.gate_max_consecutive_failures,
            s.gate_max_p95_latency_ms,
        )


@dataclass
class HealthGate:
    """Operational gate over the first ``window`` requests served by a new version.

    Fails when (a) ``max_consecutive_failures`` errors occur in a row (any sample
    count), or (b) after ``min_samples`` outcomes, error rate > ``max_error_rate`` or
    p95 latency of successful requests > ``max_p95_latency_ms``. Passes when
    ``window`` outcomes are observed without failing. These are serving-health
    signals; they say nothing about predictive quality, which needs labels.
    """

    cfg: GateConfig
    n: int = 0
    errors: int = 0
    consecutive: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, ok: bool, latency_ms: float) -> tuple[Literal["observe", "pass", "fail"], str]:
        self.n += 1
        if ok:
            self.consecutive = 0
            self.latencies_ms.append(latency_ms)
        else:
            self.errors += 1
            self.consecutive += 1
        if self.consecutive >= self.cfg.max_consecutive_failures:
            return "fail", f"{self.consecutive} consecutive failures"
        if self.n >= self.cfg.min_samples:
            rate = self.errors / self.n
            if rate > self.cfg.max_error_rate:
                return "fail", f"error rate {rate:.3f} > {self.cfg.max_error_rate} over {self.n}"
            if len(self.latencies_ms) >= self.cfg.min_samples:
                p95 = float(np.percentile(self.latencies_ms, 95))
                if p95 > self.cfg.max_p95_latency_ms:
                    return "fail", f"p95 latency {p95:.1f} ms > {self.cfg.max_p95_latency_ms}"
        if self.n >= self.cfg.window:
            return "pass", f"{self.n} requests, {self.errors} errors"
        return "observe", ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "observed": self.n,
            "errors": self.errors,
            "consecutive_failures": self.consecutive,
            "config": self.cfg.__dict__,
        }


class DriftMonitor:
    """Rolling standardized mean shift of served features vs the training reference.

    A *proxy* signal (no labels): it can say inputs look different from training
    data, never that the model got worse. It does not trigger rollback.
    """

    def __init__(self, metrics: Metrics, size: int = 500, every: int = 25) -> None:
        self.rows: deque[np.ndarray] = deque(maxlen=size)
        self.version: str | None = None
        self.metrics = metrics
        self.every = every
        self.count = 0
        self.last: dict[str, float] = {}
        self._lock = threading.Lock()

    def observe(self, lm: LoadedModel, X: pd.DataFrame) -> None:
        with self._lock:
            if lm.version != self.version:
                self.rows.clear()
                self.version = lm.version
                self.last = {}
            self.rows.extend(X.to_numpy())
            self.count += 1
            if self.count % self.every or len(self.rows) < 20:
                return
            df = pd.DataFrame(list(self.rows), columns=X.columns)
            self.last = standardized_mean_shift(df, lm.manifest.reference_stats)
        for k, v in self.last.items():
            self.metrics.feature_shift.labels(k).set(v)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.version,
                "rows": len(self.rows),
                "standardized_mean_shift": {k: round(v, 3) for k, v in self.last.items()},
                "alert_features": sorted(k for k, v in self.last.items() if v > 0.5),
                "note": "proxy signal; no labels; not a model-quality measurement",
            }


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DeploymentController:
    def __init__(
        self,
        registry: Registry,
        schema: FeatureSchema,
        settings: Settings,
        metrics: Metrics,
        limits: ResourceLimits | None = None,
    ) -> None:
        self.registry = registry
        self.schema = schema
        self.settings = settings
        self.metrics = metrics
        self.limits = limits or ResourceLimits()
        self.drift = DriftMonitor(metrics)
        self._active: LoadedModel | None = None
        self._known_good: LoadedModel | None = None
        self._gate: HealthGate | None = None
        self._promote_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self.not_ready_reason: str | None = "not started"
        self.persist_degraded: str | None = None
        # Durable write that failed after memory already changed; retried by _reconcile.
        self._pending_persist: Callable[[], None] | None = None

    # -- read side --------------------------------------------------------------------

    def current(self) -> LoadedModel | None:
        return self._active  # single attribute read: atomic under the GIL

    @property
    def on_probation(self) -> bool:
        return self._gate is not None

    def _load(self, version: str, stage: str) -> tuple[LoadedModel | None, ValidationReport]:
        rep = validate_artifact(
            self.registry.artifact_dir(version),
            self.schema,
            expected_version=version,
            limits=self.limits,
        )
        if not rep.passed or rep.manifest is None:
            self.metrics.model_load_failures.labels(stage).inc()
            return None, rep
        return LoadedModel(version, rep.manifest, rep.estimator, _now()), rep

    def _publish_active(self) -> None:
        self.metrics.active_model.clear()
        a = self._active
        if a is not None:
            self.metrics.active_model.labels(a.version, a.sha256[:16], self.schema.version).set(1)
        self.metrics.probation.set(1 if self._gate else 0)
        with contextlib.suppress(RegistryError):
            self.metrics.deployment_generation.set(self.registry.read_state()["generation"])

    def readiness(self) -> dict[str, Any]:
        a = self._active
        checks: dict[str, Any] = {"model_loaded": a is not None}
        if a is not None:
            path = self.registry.artifact_dir(a.version) / a.manifest.model_file
            try:
                checks["artifact_intact"] = sha256_file(path) == a.sha256
            except OSError:
                checks["artifact_intact"] = False
        try:
            self.registry.read_state()
            checks["registry_readable"] = True
        except RegistryError:
            checks["registry_readable"] = False
        ready = all(v for v in checks.values() if isinstance(v, bool))
        return {
            "ready": ready,
            "checks": checks,
            "reason": None if ready else (self.not_ready_reason or "check failed"),
            "model_version": a.version if a else None,
            # Serving is correct (memory holds the right model) but state.json lags it.
            # Promotions are refused until reconciled; a restart also self-heals.
            "degraded": self.persist_degraded is not None,
            "degraded_reason": self.persist_degraded,
        }

    def snapshot(self) -> dict[str, Any]:
        a, k, g = self._active, self._known_good, self._gate
        return {
            "active": a.version if a else None,
            "active_sha256": a.sha256 if a else None,
            "known_good": k.version if k else None,
            "probation": g.snapshot() if g else None,
            "persist_degraded": self.persist_degraded,
            "registry": self.registry.inspect(),
            "proxy_monitoring": self.drift.snapshot(),
        }

    # -- lifecycle --------------------------------------------------------------------

    def startup(self) -> None:
        # A version is CANDIDATE only inside a running promote(); after a restart none
        # is running, so a leftover candidate is from a crashed promotion. Release it
        # so it can be retried (it was never active, so this is not a reactivation).
        for v, e in self.registry.read_state()["versions"].items():
            if e["state"] == State.CANDIDATE:
                with contextlib.suppress(RegistryError):
                    self.registry.transition(
                        v,
                        State.VALIDATED,
                        action="candidate_released",
                        reason="restart after interrupted promotion",
                    )
        state = self.registry.read_state()
        active_v, good_v = state["active"], state["known_good"]
        if active_v is None:
            self.not_ready_reason = "no active model; run `mlpipe bootstrap <version>`"
            return
        good_lm, good_rep = (None, None) if good_v is None else self._load(good_v, "startup")
        if state.get("probation") and active_v != good_v and good_lm is None:
            # The candidate never proved itself and there is no fallback. Serving it
            # without a gate (and as its own known-good) would hide that. Fail closed.
            self.not_ready_reason = (
                f"restart during probation of {active_v}: known-good {good_v} failed to "
                f"load ({good_rep.summary()['failure_reason'] if good_rep else 'missing'}); "
                "refusing to serve an unproven candidate without a fallback. Restore the "
                "known-good artifact, then restart."
            )
            log(logger, logging.ERROR, "startup refused", reason=self.not_ready_reason)
            self._publish_active()
            return
        if state.get("probation") and good_lm is not None and active_v != good_v:
            # Restart during probation: the candidate never proved itself. Restore.
            self._known_good = good_lm
            self._active = None
            self._rollback_persist(
                active_v,
                good_lm,
                trigger="restart_during_probation",
                reason="process restarted before probation completed",
                correlation_id=uuid.uuid4().hex,
            )
            self._active = good_lm
            self.not_ready_reason = None
            self._publish_active()
            return
        lm, rep = (good_lm, good_rep) if active_v == good_v else self._load(active_v, "startup")
        if lm is None:
            if good_lm is not None and active_v != good_v:
                self._known_good = good_lm
                self._rollback_persist(
                    active_v,
                    good_lm,
                    trigger="active_load_failed_at_startup",
                    reason=(rep.summary()["failure_reason"] if rep else "load failed"),
                    correlation_id=uuid.uuid4().hex,
                )
                self._active = good_lm
                self.not_ready_reason = None
            else:
                self.not_ready_reason = (
                    f"active model {active_v} failed to load: "
                    f"{rep.summary()['failure_reason'] if rep else 'unknown'}"
                )
            self._publish_active()
            return
        self._active = lm
        self._known_good = good_lm or lm
        self.not_ready_reason = None
        self._publish_active()
        log(logger, logging.INFO, "startup", active=lm.version, known_good=self._known_good.version)

    def _reconcile(self) -> None:
        """Retry a durable write that failed after memory changed (caller holds locks)."""
        if self._pending_persist is None:
            return
        try:
            self._pending_persist()
        except Exception as e:
            self.persist_degraded = f"{self.persist_degraded} (retry failed: {e!r})"[:300]
            raise DeploymentError(
                "state_degraded",
                f"durable state lags in-memory state; reconcile failed: {e!r}"[:300],
                503,
            ) from e
        self._pending_persist = None
        self.persist_degraded = None
        self.metrics.deployment_events.labels("reconcile", "ok").inc()
        self._publish_active()

    def validate(self, version: str, *, correlation_id: str | None = None) -> dict[str, Any]:
        cid = correlation_id or uuid.uuid4().hex
        st = self.registry.state_of(version)
        if st is None:
            raise DeploymentError("unknown_version", version, 404)
        rep = validate_artifact(
            self.registry.artifact_dir(version),
            self.schema,
            expected_version=version,
            limits=self.limits,
        )
        summary = rep.summary()
        if st == State.REGISTERED:
            to = State.VALIDATED if rep.passed else State.REJECTED
            self.registry.transition(
                version,
                to,
                action="validate",
                correlation_id=cid,
                validation=summary,
                failure_reason=summary["failure_reason"],
            )
            self.metrics.deployment_events.labels(
                "validate", "passed" if rep.passed else "rejected"
            ).inc()
            if not rep.passed:
                self.metrics.model_load_failures.labels("validate").inc()
        return {**summary, "state": str(self.registry.state_of(version))}

    def bootstrap(self, version: str, *, correlation_id: str | None = None) -> dict[str, Any]:
        cid = correlation_id or uuid.uuid4().hex
        with self._promote_lock, self._state_lock:
            state = self.registry.read_state()
            if state["active"] is not None:
                raise DeploymentError("already_bootstrapped", f"active is {state['active']}")
            res = self.validate(version, correlation_id=cid)
            if not res["passed"]:
                raise DeploymentError("validation_failed", str(res["failure_reason"]), 422)
            lm, rep = self._load(version, "bootstrap")
            if lm is None:
                raise DeploymentError(
                    "validation_failed", str(rep.summary()["failure_reason"]), 422
                )
            self.registry.apply_many(
                [(version, State.ACTIVE)],
                action="bootstrap",
                correlation_id=cid,
                pointer_updates={"active": version, "known_good": version, "probation": None},
                validation=rep.summary(),
            )
            self._active = self._known_good = lm
            self._gate = None
            self.not_ready_reason = None
            self._publish_active()
            self.metrics.deployment_events.labels("bootstrap", "ok").inc()
            return {"result": "active", "version": version}

    def promote(
        self,
        version: str,
        *,
        expected_generation: int | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        cid = correlation_id or uuid.uuid4().hex
        if not self._promote_lock.acquire(blocking=False):
            self.metrics.deployment_events.labels("promote", "busy").inc()
            raise DeploymentError("promotion_in_progress", "another promotion is running")
        try:
            return self._promote(version, expected_generation, cid)
        finally:
            self._promote_lock.release()

    def _promote(self, version: str, expected_generation: int | None, cid: str) -> dict[str, Any]:
        with self._state_lock:
            self._reconcile()
        state = self.registry.read_state()
        if expected_generation is not None and expected_generation != state["generation"]:
            self.metrics.deployment_events.labels("promote", "stale").inc()
            raise DeploymentError(
                "stale_generation",
                f"expected generation {expected_generation}, is {state['generation']}",
            )
        if self._active is None or self._known_good is None:
            raise DeploymentError("not_bootstrapped", "no active model", 409)
        if self._gate is not None:
            raise DeploymentError("probation_in_progress", f"{self._active.version} on probation")
        st = self.registry.state_of(version)
        if st is None:
            raise DeploymentError("unknown_version", version, 404)
        if st == State.REGISTERED:
            if not self.validate(version, correlation_id=cid)["passed"]:
                self.metrics.deployment_events.labels("promote", "rejected").inc()
                return {
                    "result": "rejected",
                    "version": version,
                    "validation": validate_artifact(
                        self.registry.artifact_dir(version), self.schema
                    ).summary(),
                }
            st = State.VALIDATED
        if st != State.VALIDATED:
            raise DeploymentError("invalid_state", f"{version} is {st}; must be validated")

        self.registry.transition(
            version, State.CANDIDATE, action="select_candidate", correlation_id=cid
        )
        lm, rep = self._load(version, "promote")
        if lm is not None:
            twin = self.registry.known_bad_shas(self.registry.read_state()).get(lm.sha256)
            if twin is not None:
                from mlpipe.validation import Check

                rep.checks.append(Check("not_known_bad", False, f"identical bytes to {twin}"))
                lm = None
        if lm is None:
            summary = rep.summary()
            self.registry.transition(
                version,
                State.REJECTED,
                action="promote_rejected",
                correlation_id=cid,
                validation=summary,
                failure_reason=summary["failure_reason"],
            )
            self.metrics.deployment_events.labels("promote", "rejected").inc()
            log(
                logger,
                logging.WARNING,
                "promotion rejected",
                version=version,
                reason=summary["failure_reason"],
                correlation_id=cid,
            )
            return {"result": "rejected", "version": version, "validation": summary}

        prev = self._active
        try:
            out = self.registry.apply_many(
                [(version, State.ACTIVE), (prev.version, State.RETIRED)],
                action="promote",
                correlation_id=cid,
                pointer_updates={
                    "active": version,
                    "known_good": self._known_good.version,
                    "probation": {
                        "version": version,
                        "started_at": _now(),
                        "gate": GateConfig.from_settings(self.settings).__dict__,
                    },
                },
                validation=rep.summary(),
            )
        except Exception as e:
            landed = False
            with contextlib.suppress(Exception):
                after = self.registry.read_state()
                landed = (
                    after["active"] == version
                    and (after.get("probation") or {}).get("version") == version
                )
            if landed:
                # The commit reached state.json and the error came after it (e.g. dir
                # fsync). Disk says the candidate is on probation: make memory agree.
                log(
                    logger,
                    logging.WARNING,
                    "promote commit landed despite error",
                    version=version,
                    error=repr(e)[:200],
                )
                out = {"state": after}
            else:
                self._abort_promotion(version, cid, e)
        crashpoint("promote:after_commit_before_activate")
        with self._state_lock:
            self._gate = HealthGate(GateConfig.from_settings(self.settings))
            self._active = lm
        crashpoint("promote:after_activate")
        self._publish_active()
        self.metrics.deployment_events.labels("promote", "ok").inc()
        log(
            logger,
            logging.INFO,
            "promoted",
            version=version,
            previous=prev.version,
            known_good=self._known_good.version,
            correlation_id=cid,
        )
        return {
            "result": "active_on_probation",
            "version": version,
            "previous": prev.version,
            "known_good": self._known_good.version,
            "generation": out["state"]["generation"],
        }

    def _abort_promotion(self, version: str, cid: str, e: Exception) -> NoReturn:
        """Interrupted promotion: nothing in memory changed; put the version back."""
        self.metrics.deployment_events.labels("promote", "failed").inc()
        with contextlib.suppress(Exception):
            self.registry.transition(
                version,
                State.VALIDATED,
                action="promote_aborted",
                correlation_id=cid,
                failure_reason=repr(e)[:300],
            )
        raise DeploymentError("promotion_failed", f"state commit failed: {e!r}", 500) from e

    # -- health gate / rollback ---------------------------------------------------------

    def record_outcome(self, lm: LoadedModel, ok: bool, latency_ms: float) -> None:
        if self._gate is None:
            return
        with self._state_lock:
            gate = self._gate
            if gate is None or lm is not self._active:
                return  # outcome from a model that is no longer the probation subject
            decision, why = gate.record(ok, latency_ms)
            if decision == "fail":
                self._rollback_locked(
                    trigger="health_gate",
                    reason=why,
                    correlation_id=uuid.uuid4().hex,
                    gate=gate.snapshot(),
                )
            elif decision == "pass":
                self._finish_probation(lm, why, gate)

    def _finish_probation(self, lm: LoadedModel, why: str, gate: HealthGate) -> None:
        snap = gate.snapshot()

        def persist() -> None:
            self.registry.apply_many(
                [(lm.version, State.ACTIVE)],
                action="probation_passed",
                pointer_updates={"known_good": lm.version, "probation": None},
                gate=snap,
                reason=why,
            )

        try:
            persist()
        except Exception as e:
            # Safe direction: a restart would roll back a candidate that actually passed.
            self.persist_degraded = f"probation_passed not persisted: {e!r}"[:300]
            self._pending_persist = persist
        self._known_good = lm
        self._gate = None
        self._publish_active()
        self.metrics.deployment_events.labels("probation", "passed").inc()

    def rollback(
        self, *, reason: str, trigger: str = "manual", correlation_id: str | None = None
    ) -> dict[str, Any]:
        with self._state_lock:
            return self._rollback_locked(
                trigger=trigger, reason=reason, correlation_id=correlation_id or uuid.uuid4().hex
            )

    def _rollback_locked(
        self, *, trigger: str, reason: str, correlation_id: str, **details: Any
    ) -> dict[str, Any]:
        bad, good = self._active, self._known_good
        if bad is None or good is None or self._gate is None or bad is good:
            self.metrics.deployment_events.labels("rollback", "noop").inc()
            return {
                "result": "noop",
                "active": bad.version if bad else None,
                "reason": "nothing on probation; active is known-good",
            }
        # Memory first: stop serving the bad version even if persistence fails.
        self._active = good
        self._gate = None
        crashpoint("rollback:after_swap")
        self._rollback_persist(
            bad.version,
            good,
            trigger=trigger,
            reason=reason,
            correlation_id=correlation_id,
            **details,
        )
        self._publish_active()
        self.metrics.rollbacks.labels(trigger).inc()
        log(
            logger,
            logging.WARNING,
            "rolled back",
            bad=bad.version,
            restored=good.version,
            trigger=trigger,
            reason=reason,
            correlation_id=correlation_id,
        )
        return {
            "result": "rolled_back",
            "from": bad.version,
            "to": good.version,
            "trigger": trigger,
            "reason": reason,
        }

    def _rollback_persist(
        self,
        bad_version: str,
        good: LoadedModel,
        *,
        trigger: str,
        reason: str,
        correlation_id: str,
        **details: Any,
    ) -> None:
        verified = self._verify_recovery(good)

        def persist() -> None:
            self.registry.apply_many(
                [(bad_version, State.ROLLED_BACK), (good.version, State.ACTIVE)],
                action="rollback",
                correlation_id=correlation_id,
                pointer_updates={
                    "active": good.version,
                    "known_good": good.version,
                    "probation": None,
                },
                rollback_trigger=trigger,
                failure_reason=reason,
                recovery_result="verified" if verified else "smoke_failed",
                **details,
            )

        try:
            persist()
            self.persist_degraded = None
            self._pending_persist = None
        except Exception as e:
            # Memory already serves known-good. state.json still says the bad version is
            # active on probation, so a restart also restores known-good.
            self.persist_degraded = f"rollback not persisted: {e!r}"[:300]
            self._pending_persist = persist
            log(logger, logging.ERROR, "rollback persist failed", error=repr(e)[:200])
        self.metrics.deployment_events.labels(
            "rollback", "ok" if verified else "smoke_failed"
        ).inc()

    def _verify_recovery(self, lm: LoadedModel) -> bool:
        from mlpipe.features import transform
        from mlpipe.validation import check_outputs, smoke_frame

        try:
            X = transform(smoke_frame().head(4), self.schema)
            return check_outputs(lm.estimator.predict_proba(X), len(X)) is None
        except Exception:
            return False
