"""Prometheus metrics, structured JSON logging and request-ID propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Never log these keys' values even if a caller passes them in ``extra``.
_REDACT = {"features", "payload", "body", "admin_token", "authorization", "rows"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            out["request_id"] = rid
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                out[k] = "[redacted]" if k.lower() in _REDACT else v
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(out, default=str)


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Always writes to the *current* sys.stdout (survives stream redirection)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def configure_logging(level: str) -> None:
    handler = _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("mlpipe")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


class Metrics:
    """One registry per app instance (tests create many apps in one process)."""

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "mlpipe_requests_total", "HTTP requests", ["route", "status"], registry=r
        )
        self.predictions = Counter(
            "mlpipe_predictions_total", "Successful predictions", ["model_version"], registry=r
        )
        self.errors = Counter(
            "mlpipe_prediction_errors_total",
            "Failed prediction attempts",
            ["model_version", "reason"],
            registry=r,
        )
        self.latency = Histogram(
            "mlpipe_request_latency_seconds",
            "End-to-end predict latency",
            ["route"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=r,
        )
        self.in_flight = Gauge("mlpipe_in_flight_requests", "Admitted predict requests", registry=r)
        self.rejected = Counter(
            "mlpipe_admission_rejected_total",
            "Requests refused at admission",
            ["reason"],
            registry=r,
        )
        self.model_load_failures = Counter(
            "mlpipe_model_load_failures_total",
            "Model load/validation failures",
            ["stage"],
            registry=r,
        )
        self.feature_validation_failures = Counter(
            "mlpipe_feature_validation_failures_total", "Rejected request payloads", registry=r
        )
        self.audit_failures = Counter(
            "mlpipe_audit_log_failures_total", "Prediction audit write failures", registry=r
        )
        self.deployment_events = Counter(
            "mlpipe_deployment_events_total", "Deployment actions", ["action", "result"], registry=r
        )
        self.rollbacks = Counter("mlpipe_rollbacks_total", "Rollbacks", ["trigger"], registry=r)
        self.active_model = Gauge(
            "mlpipe_active_model_info",
            "1 for the model version currently serving",
            ["model_version", "model_sha256", "schema_version"],
            registry=r,
        )
        self.deployment_generation = Gauge(
            "mlpipe_deployment_generation", "Committed registry generation", registry=r
        )
        self.probation = Gauge(
            "mlpipe_probation_active", "1 while a promoted version is on probation", registry=r
        )
        self.feature_shift = Gauge(
            "mlpipe_feature_mean_shift",
            "Standardized mean shift of recent served features vs training reference (proxy)",
            ["feature"],
            registry=r,
        )
