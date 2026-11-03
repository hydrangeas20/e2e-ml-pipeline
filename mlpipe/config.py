"""Environment-based, validated service configuration (prefix ``MLPIPE_``).

Invalid configuration fails at startup with a readable error, never at first request.
``config_version`` is a hash of every non-secret setting and is attached to every
prediction so a response can be traced to the configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLPIPE_", extra="forbid")

    registry_dir: Path = Path("registry")
    audit_log_path: Path = Path("registry/predictions.jsonl")
    audit_required: bool = True
    audit_fsync: bool = False

    # admission / concurrency
    max_concurrency: int = Field(4, ge=1, le=64)
    max_queue: int = Field(16, ge=0, le=10_000)
    request_timeout_s: float = Field(2.0, gt=0, le=60)
    drain_timeout_s: float = Field(10.0, ge=0, le=300)
    max_batch_rows: int = Field(64, ge=1, le=1024)

    # post-promotion health gate (applies only to the version on probation)
    gate_min_samples: int = Field(20, ge=1)
    gate_window: int = Field(100, ge=1)
    gate_max_error_rate: float = Field(0.10, ge=0, le=1)
    gate_max_consecutive_failures: int = Field(5, ge=1)
    gate_max_p95_latency_ms: float = Field(250.0, gt=0)

    enable_fault_injection: bool = False
    admin_token: SecretStr | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.gate_min_samples > self.gate_window:
            raise ValueError("gate_min_samples must be <= gate_window")
        if self.admin_token is not None and len(self.admin_token.get_secret_value()) < 16:
            raise ValueError("admin_token must be at least 16 characters")
        return self

    @property
    def config_version(self) -> str:
        public = self.model_dump(mode="json", exclude={"admin_token"})
        public["admin_enabled"] = self.admin_token is not None
        canon = json.dumps(public, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]
