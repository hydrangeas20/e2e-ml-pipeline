"""Filesystem model registry, deployment state machine and deployment event journal.

Layout::

    <root>/
      models/<version>/{model.skops,manifest.json}   immutable once registered
      state.json                                     source of truth (atomic replace)
      events.jsonl                                   append-only journal (fsync'd)
      .lock                                          fcntl lock serialising writers

Durability: ``state.json`` is replaced atomically (write temp, fsync, rename, fsync
dir). Each event is appended and fsync'd *before* the state it describes is
committed, so a crash between the two leaves an event whose ``generation`` exceeds
the committed generation; ``inspect`` reports these as ``uncommitted``. Single host
only: fcntl locks do not work across machines or on most network filesystems.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from mlpipe.artifacts import VERSION_RE, ArtifactError, read_manifest, write_json_atomic


class State(StrEnum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    CANDIDATE = "candidate"  # selected for promotion; being loaded/smoke-tested in-server
    ACTIVE = "active"  # serving traffic (on probation until the health gate passes)
    RETIRED = "retired"  # previously active and healthy; eligible as known-good
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


TRANSITIONS: dict[State, frozenset[State]] = {
    State.REGISTERED: frozenset({State.VALIDATED, State.REJECTED}),
    State.VALIDATED: frozenset({State.CANDIDATE, State.REJECTED, State.ACTIVE}),
    State.CANDIDATE: frozenset({State.ACTIVE, State.REJECTED, State.VALIDATED}),
    State.ACTIVE: frozenset({State.RETIRED, State.ROLLED_BACK}),
    State.RETIRED: frozenset({State.ACTIVE}),  # rollback target
    State.REJECTED: frozenset(),
    State.ROLLED_BACK: frozenset(),
}


class RegistryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class StaleGenerationError(RegistryError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__("stale_generation", f"expected generation {expected}, is {actual}")
        self.expected = expected
        self.actual = actual


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _empty_state() -> dict[str, Any]:
    return {
        "generation": 0,
        "active": None,
        "known_good": None,
        "probation": None,
        "versions": {},
    }


class Registry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self._lock_path = self.root / ".lock"
        self._tlock = threading.RLock()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # -- locking / persistence ------------------------------------------------------

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        with self._tlock, self._lock_path.open("a") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return _empty_state()
        try:
            state: dict[str, Any] = json.loads(self.state_path.read_text())
        except json.JSONDecodeError as e:
            raise RegistryError("state_corrupt", f"state.json unreadable: {e}") from e
        return state

    def _commit(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        record = {"event_id": uuid.uuid4().hex, "ts": _now(), **event}
        line = json.dumps(record, sort_keys=True) + "\n"
        fd = os.open(self.events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return record

    def events(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out = []
        for line in self.events_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"action": "unparseable_event_line"})
        return out[-limit:] if limit else out

    # -- queries --------------------------------------------------------------------

    def artifact_dir(self, version: str) -> Path:
        if not VERSION_RE.match(version):
            raise RegistryError("invalid_version", f"bad version {version!r}")
        return self.models_dir / version

    def state_of(self, version: str) -> State | None:
        v = self.read_state()["versions"].get(version)
        return State(v["state"]) if v else None

    def inspect(self) -> dict[str, Any]:
        state = self.read_state()
        uncommitted = [
            e for e in self.events() if int(e.get("generation", 0)) > int(state["generation"])
        ]
        return {**state, "uncommitted_events": uncommitted}

    # -- mutations ------------------------------------------------------------------

    def register(self, src_dir: Path, *, correlation_id: str | None = None) -> str:
        manifest = read_manifest(src_dir)
        version = manifest.model_version
        dest = self.artifact_dir(version)
        with self.locked():
            state = self.read_state()
            if version in state["versions"] or dest.exists():
                raise RegistryError("duplicate_version", f"{version} already registered")
            tmp = Path(tempfile.mkdtemp(dir=self.models_dir, prefix=f".{version}."))
            try:
                for name in ("model.skops", "manifest.json"):
                    if (src_dir / name).exists():
                        shutil.copy2(src_dir / name, tmp / name)
                os.replace(tmp, dest)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            self._apply(
                state,
                version,
                None,
                State.REGISTERED,
                action="register",
                correlation_id=correlation_id,
            )
        return version

    def transition(
        self,
        version: str,
        to: State,
        *,
        action: str,
        expected_generation: int | None = None,
        correlation_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        with self.locked():
            state = self.read_state()
            if expected_generation is not None and expected_generation != state["generation"]:
                raise StaleGenerationError(expected_generation, state["generation"])
            entry = state["versions"].get(version)
            if entry is None:
                raise RegistryError("unknown_version", version)
            return self._apply(
                state,
                version,
                State(entry["state"]),
                to,
                action=action,
                correlation_id=correlation_id,
                **details,
            )

    def _apply(
        self,
        state: dict[str, Any],
        version: str,
        frm: State | None,
        to: State,
        *,
        action: str,
        correlation_id: str | None,
        pointer_updates: dict[str, Any] | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        if frm is not None and to not in TRANSITIONS[frm]:
            raise RegistryError("invalid_transition", f"{version}: {frm} -> {to} not allowed")
        new = json.loads(json.dumps(state))
        new["generation"] = int(state["generation"]) + 1
        new["versions"][version] = {"state": str(to), "updated_at": _now(), "action": action}
        for k, v in (pointer_updates or {}).items():
            new[k] = v
        event = self.append_event(
            {
                "correlation_id": correlation_id or uuid.uuid4().hex,
                "generation": new["generation"],
                "model_version": version,
                "action": action,
                "from_state": str(frm) if frm else None,
                "to_state": str(to),
                **details,
            }
        )
        self._commit(new)
        return {"state": new, "event": event}

    def apply_many(
        self,
        changes: list[tuple[str, State]],
        *,
        action: str,
        pointer_updates: dict[str, Any],
        expected_generation: int | None = None,
        correlation_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        """Several version transitions + pointer moves committed as ONE generation."""
        with self.locked():
            state = self.read_state()
            if expected_generation is not None and expected_generation != state["generation"]:
                raise StaleGenerationError(expected_generation, state["generation"])
            new = json.loads(json.dumps(state))
            new["generation"] = int(state["generation"]) + 1
            transitions = []
            for version, to in changes:
                entry = state["versions"].get(version)
                if entry is None:
                    raise RegistryError("unknown_version", version)
                frm = State(entry["state"])
                if frm != to and to not in TRANSITIONS[frm]:
                    raise RegistryError(
                        "invalid_transition", f"{version}: {frm} -> {to} not allowed"
                    )
                new["versions"][version] = {"state": str(to), "updated_at": _now(), "action": action}
                transitions.append({"model_version": version, "from_state": str(frm), "to_state": str(to)})
            new.update(pointer_updates)
            event = self.append_event(
                {
                    "correlation_id": correlation_id or uuid.uuid4().hex,
                    "generation": new["generation"],
                    "action": action,
                    "model_version": changes[0][0] if changes else None,
                    "from_state": transitions[0]["from_state"] if transitions else None,
                    "to_state": transitions[0]["to_state"] if transitions else None,
                    "transitions": transitions,
                    "previous_active": state["active"],
                    "next_active": new["active"],
                    **details,
                }
            )
            self._commit(new)
            return {"state": new, "event": event}


def ensure_version_dir(registry: Registry, version: str) -> Path:
    d = registry.artifact_dir(version)
    if not d.is_dir():
        raise ArtifactError("model_missing", f"no artifact directory for {version}")
    return d
