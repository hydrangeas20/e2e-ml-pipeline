"""FastAPI serving application (the single inference API; ``app/main.py`` re-exports it).

Request admission
-----------------
``max_concurrency`` worker threads run inference; up to ``max_queue`` further
requests may wait for a worker. Anything beyond ``max_concurrency + max_queue`` is
refused immediately with 429. Admission slots are released when the worker thread
finishes, not when the client gives up, so a timed-out request still occupies its
slot until its computation ends: resource use stays bounded.

Timeout enforcement limitation: ``request_timeout_s`` bounds how long a *client*
waits (504). Python cannot interrupt a running thread, so the computation itself is
not cancelled; it runs to completion and its result is discarded.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from mlpipe.config import Settings
from mlpipe.features import FeatureValidationError, transform
from mlpipe.registry import Registry, RegistryError
from mlpipe.schema import SERVING_SCHEMA
from mlpipe.serving.audit import AuditError, PredictionAuditLog
from mlpipe.serving.deployment import DeploymentController, DeploymentError, LoadedModel
from mlpipe.serving.faults import Fault, FaultInjector, FaultKind, InjectedFault
from mlpipe.serving.observability import Metrics, configure_logging, log, request_id_var
from mlpipe.validation import check_outputs

logger = logging.getLogger("mlpipe.http")
_RID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_I = {"strict": True}
_F = {"strict": True, "allow_inf_nan": False}


class Features(BaseModel):
    """gdm-features/1. Unknown fields, strings, booleans, NaN/inf are rejected."""

    model_config = ConfigDict(extra="forbid")

    Pregnancies: Annotated[int, Field(ge=0, le=30, **_I)]
    Glucose: Annotated[int, Field(ge=0, le=400, **_I)]
    BloodPressure: Annotated[int, Field(ge=0, le=250, **_I)]
    SkinThickness: Annotated[int, Field(ge=0, le=120, **_I)]
    Insulin: Annotated[int, Field(ge=0, le=1000, **_I)]
    BMI: Annotated[float, Field(ge=0, le=90, **_F)]
    DiabetesPedigreeFunction: Annotated[float, Field(ge=0, le=3, **_F)]
    Age: Annotated[int, Field(ge=0, le=120, **_I)]


assert list(Features.model_fields) == SERVING_SCHEMA.names, "request model drifted from schema"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Features] = Field(min_length=1)


class Prediction(BaseModel):
    prediction: int
    probability: float


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    model_sha256: str
    schema_version: str
    config_version: str
    decision_threshold: float
    predictions: list[Prediction]


class LegacyPredictResponse(BaseModel):
    prediction: int
    confidence: float
    probability: float
    request_id: str
    model_version: str
    model_sha256: str
    config_version: str


class PromoteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    expected_generation: int | None = None


class RollbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field("manual", max_length=200)


class FaultBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: FaultKind
    model_version: str | None = None
    n: int = Field(0, ge=0)
    defect: str | None = None
    seconds: float = 0.0


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        self.status, self.code, self.message, self.details = status, code, message, details


def _err(status: int, code: str, message: str, **details: Any) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id_var.get(),
            **({"details": details} if details else {}),
        }
    }
    return JSONResponse(body, status_code=status)


class InvalidOutput(RuntimeError):
    pass


class _Ticket:
    """Per-request handshake between the worker thread and the waiting handler.

    Exactly one of these happens, under ``lock``: the worker commits (audit written,
    result will be returned) or the handler abandons (client got 504; the worker
    must not audit or record the result). This keeps the audit log truthful.
    """

    __slots__ = ("abandoned", "committed", "lock")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.abandoned = False
        self.committed = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    metrics = Metrics()
    registry = Registry(settings.registry_dir)
    controller = DeploymentController(registry, SERVING_SCHEMA, settings, metrics)
    faults = FaultInjector(settings.enable_fault_injection)
    audit = PredictionAuditLog(
        settings.audit_log_path, fsync=settings.audit_fsync, fail_hook=faults.audit_should_fail
    )
    executor = ThreadPoolExecutor(max_workers=settings.max_concurrency, thread_name_prefix="infer")
    capacity = settings.max_concurrency + settings.max_queue
    lifecycle: dict[str, Any] = {"admitted": 0, "draining": False, "started": False}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        serve_lock = registry.acquire_serve_lock()  # raises if another server owns it
        try:
            async with _lifespan_body():
                yield
        finally:
            serve_lock.close()

    @asynccontextmanager
    async def _lifespan_body() -> AsyncIterator[None]:
        controller.startup()
        lifecycle["started"] = True
        log(
            logger,
            logging.INFO,
            "service started",
            config_version=settings.config_version,
            ready=controller.readiness()["ready"],
        )
        yield
        lifecycle["draining"] = True
        deadline = time.monotonic() + settings.drain_timeout_s
        while lifecycle["admitted"] > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        log(logger, logging.INFO, "service stopped", abandoned=lifecycle["admitted"])
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Diabetes Prediction API", version="2.0.0", lifespan=lifespan)
    app.state.controller = controller
    app.state.faults = faults
    app.state.metrics = metrics
    app.state.lifecycle = lifecycle
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.requests.labels(path, str(response.status_code)).inc()
        token = request_id_var.set(rid)
        log(
            logger,
            logging.INFO,
            "request",
            method=request.method,
            route=path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        request_id_var.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        if request.url.path in ("/predict", "/v1/predict"):
            metrics.feature_validation_failures.inc()
        # Echo location and message only: never the offending input values.
        details = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _err(422, "invalid_request", "request failed schema validation", errors=details)

    @app.exception_handler(ServiceError)
    async def on_service(_: Request, exc: ServiceError) -> JSONResponse:
        return _err(exc.status, exc.code, exc.message, **exc.details)

    @app.exception_handler(DeploymentError)
    async def on_deploy(_: Request, exc: DeploymentError) -> JSONResponse:
        return _err(exc.status, exc.code, exc.message)

    @app.exception_handler(RegistryError)
    async def on_registry(_: Request, exc: RegistryError) -> JSONResponse:
        return _err(409, exc.code, exc.message)

    # -- inference ------------------------------------------------------------------

    def infer(
        lm: LoadedModel, rows: list[dict[str, Any]], rid: str, ticket: _Ticket
    ) -> list[Prediction] | None:
        faults.before_predict(lm.version)
        X = transform(pd.DataFrame(rows), SERVING_SCHEMA, defect=faults.feature_defect(lm.version))
        proba = lm.estimator.predict_proba(X)
        bad = check_outputs(proba, len(rows))
        if bad:
            raise InvalidOutput(bad)
        p1 = [float(p) for p in proba[:, 1]]
        preds = [Prediction(prediction=int(p >= lm.threshold), probability=p) for p in p1]
        with ticket.lock:
            if ticket.abandoned:
                return None  # client already got 504: do not audit or count this result
            _audit(lm, preds, rid)
            ticket.committed = True
        controller.drift.observe(lm, X)
        return preds

    def _audit(lm: LoadedModel, preds: list[Prediction], rid: str) -> None:
        try:
            audit.write(
                [
                    {
                        "request_id": rid,
                        "row": i,
                        "model_version": lm.version,
                        "model_sha256": lm.sha256,
                        "config_version": settings.config_version,
                        "prediction": pr.prediction,
                        "probability": round(pr.probability, 6),
                    }
                    for i, pr in enumerate(preds)
                ]
            )
        except AuditError:
            metrics.audit_failures.inc()
            if settings.audit_required:
                raise
            log(logger, logging.WARNING, "audit write failed (not required)")

    async def serve(rows: list[dict[str, Any]], route: str) -> tuple[LoadedModel, list[Prediction]]:
        if lifecycle["draining"]:
            metrics.rejected.labels("draining").inc()
            raise ServiceError(503, "draining", "service is shutting down")
        lm = controller.current()
        if lm is None:
            metrics.rejected.labels("not_ready").inc()
            raise ServiceError(503, "not_ready", controller.not_ready_reason or "no model")
        if len(rows) > settings.max_batch_rows:
            metrics.feature_validation_failures.inc()
            raise ServiceError(422, "batch_too_large", f"max {settings.max_batch_rows} instances")
        if lifecycle["admitted"] >= capacity:
            metrics.rejected.labels("overloaded").inc()
            raise ServiceError(
                429, "overloaded", "admission limit reached; retry later", capacity=capacity
            )
        lifecycle["admitted"] += 1
        metrics.in_flight.inc()
        loop = asyncio.get_running_loop()
        rid = request_id_var.get() or uuid.uuid4().hex
        t0 = time.perf_counter()
        ticket = _Ticket()
        fut = loop.run_in_executor(executor, infer, lm, rows, rid, ticket)

        def release(_: object) -> None:
            lifecycle["admitted"] -= 1
            metrics.in_flight.dec()

        fut.add_done_callback(release)
        try:
            try:
                result = await asyncio.wait_for(asyncio.shield(fut), settings.request_timeout_s)
            except TimeoutError:
                with ticket.lock:
                    late_commit = ticket.committed
                    ticket.abandoned = not late_commit
                if not late_commit:
                    raise
                result = await fut  # audit already written: the result must be returned
        except TimeoutError:

            def on_abandoned_done(f: asyncio.Future[Any]) -> None:
                if f.cancelled():
                    metrics.abandoned.labels("cancelled").inc()
                elif f.exception() is not None:
                    metrics.abandoned.labels("failed").inc()
                else:
                    metrics.abandoned.labels("completed").inc()

            fut.add_done_callback(on_abandoned_done)
            await fail(lm, "timeout", t0)
            raise ServiceError(
                504,
                "timeout",
                f"inference exceeded {settings.request_timeout_s}s",
                model_version=lm.version,
            ) from None
        except AuditError:
            metrics.errors.labels(lm.version, "audit").inc()
            raise ServiceError(
                503,
                "audit_unavailable",
                "prediction not returned: audit record could not be written",
                model_version=lm.version,
            ) from None
        except (InjectedFault, InvalidOutput, FeatureValidationError) as e:
            reason = {
                InjectedFault: "injected",
                InvalidOutput: "invalid_output",
                FeatureValidationError: "feature_pipeline",
            }[type(e)]
            await fail(lm, reason, t0)
            raise ServiceError(500, "prediction_failed", reason, model_version=lm.version) from None
        except Exception as e:
            await fail(lm, "internal", t0)
            log(logger, logging.ERROR, "inference error", error_type=type(e).__name__)
            raise ServiceError(
                500, "prediction_failed", "internal", model_version=lm.version
            ) from None
        assert result is not None  # only None when abandoned, which raised above
        preds = result
        latency = (time.perf_counter() - t0) * 1000
        metrics.latency.labels(route).observe(latency / 1000)
        metrics.predictions.labels(lm.version).inc(len(preds))
        # Gate bookkeeping may roll back (fsync'd I/O): keep it off the event loop.
        await asyncio.to_thread(controller.record_outcome, lm, True, latency)
        return lm, preds

    async def fail(lm: LoadedModel, reason: str, t0: float) -> None:
        metrics.errors.labels(lm.version, reason).inc()
        latency = (time.perf_counter() - t0) * 1000
        await asyncio.to_thread(controller.record_outcome, lm, False, latency)

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict_v1(body: PredictRequest) -> PredictResponse:
        lm, preds = await serve([i.model_dump() for i in body.instances], "/v1/predict")
        return PredictResponse(
            request_id=request_id_var.get() or "",
            model_version=lm.version,
            model_sha256=lm.sha256,
            schema_version=SERVING_SCHEMA.version,
            config_version=settings.config_version,
            decision_threshold=lm.threshold,
            predictions=preds,
        )

    @app.post("/predict", response_model=LegacyPredictResponse)
    async def predict_legacy(body: Features) -> LegacyPredictResponse:
        """Backward-compatible with the original API: flat payload, prediction+confidence."""
        lm, preds = await serve([body.model_dump()], "/predict")
        p = preds[0]
        return LegacyPredictResponse(
            prediction=p.prediction,
            confidence=max(p.probability, 1 - p.probability),
            probability=p.probability,
            request_id=request_id_var.get() or "",
            model_version=lm.version,
            model_sha256=lm.sha256,
            config_version=settings.config_version,
        )

    # -- health ---------------------------------------------------------------------

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok"}  # original endpoint, kept for compatibility

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        r = await asyncio.to_thread(controller.readiness)  # re-hashes the artifact
        if lifecycle["draining"]:
            r = {**r, "ready": False, "reason": "draining"}
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/health")
    async def health() -> JSONResponse:
        return await readyz()

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # -- admin ----------------------------------------------------------------------

    def admin(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.admin_token is None:
            raise ServiceError(403, "admin_disabled", "set MLPIPE_ADMIN_TOKEN to enable")
        expected = f"Bearer {settings.admin_token.get_secret_value()}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise ServiceError(401, "unauthorized", "missing or invalid admin token")

    adm = [Depends(admin)]

    @app.get("/admin/deployment", dependencies=adm)
    async def deployment() -> dict[str, Any]:
        return controller.snapshot()

    @app.get("/admin/events", dependencies=adm)
    async def events(limit: int = 50) -> list[dict[str, Any]]:
        return registry.events(limit=min(max(limit, 1), 1000))

    @app.post("/admin/validate/{version}", dependencies=adm)
    async def validate(version: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            controller.validate, version, correlation_id=request_id_var.get()
        )

    @app.post("/admin/promote", dependencies=adm)
    async def promote(body: PromoteBody) -> JSONResponse:
        rid = request_id_var.get()
        res = await asyncio.to_thread(
            controller.promote,
            body.version,
            expected_generation=body.expected_generation,
            correlation_id=rid,
        )
        return JSONResponse(res, status_code=200 if res["result"] != "rejected" else 422)

    @app.post("/admin/rollback", dependencies=adm)
    async def rollback(body: RollbackBody) -> dict[str, Any]:
        return await asyncio.to_thread(
            controller.rollback, reason=body.reason, correlation_id=request_id_var.get()
        )

    @app.get("/admin/faults", dependencies=adm)
    async def list_faults() -> list[dict[str, Any]]:
        return faults.list()

    @app.post("/admin/faults", dependencies=adm)
    async def set_fault(body: FaultBody) -> list[dict[str, Any]]:
        try:
            faults.set(Fault(**body.model_dump()))
        except PermissionError as e:
            raise ServiceError(403, "faults_disabled", str(e)) from None
        except ValueError as e:
            raise ServiceError(422, "invalid_fault", str(e)) from None
        log(logger, logging.WARNING, "fault injected", **body.model_dump())
        return faults.list()

    @app.delete("/admin/faults", dependencies=adm)
    async def clear_faults() -> list[dict[str, Any]]:
        faults.clear()
        return []

    return app
