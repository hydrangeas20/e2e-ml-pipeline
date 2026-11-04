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
from mlpipe.crash import crashpoint


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

    def acquire_serve_lock(self) -> Any:
        """Exclusive, non-blocking lock held by the one serving process for this registry.

        Active-model state lives in process memory, so two servers (or ``uvicorn
        --workers N``) on one registry would each run their own probation and rollback.
        Returns an open file whose lifetime is the lock; close it to release.
        """
        f = (self.root / ".serve.lock").open("a+")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            holder = f.read().strip() or "unknown"
            f.close()
            raise RuntimeError(
                f"serving lock on {self.root} is held by pid {holder}: only one server "
                "process per registry is supported (do not use --workers > 1)"
            ) from None
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        return f

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

    def _last_journal_generation(self) -> int:
        """Highest ``generation`` in the journal tail (0 if none). Called under the lock."""
        if not self.events_path.exists():
            return 0
        with self.events_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode(errors="replace").splitlines()
        for line in reversed(tail):
            try:
                gen = json.loads(line).get("generation")
            except (json.JSONDecodeError, AttributeError):
                continue
            if isinstance(gen, int):
                return gen
        return 0

    def _next_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Copy of ``state`` with a fresh, never-before-journaled generation.

        If an earlier writer journaled an event and then failed (or crashed) before
        committing it, the journal is ahead of state.json. Those generations are
        recorded as aborted so they are never reused and stay visible as uncommitted.
        """
        new: dict[str, Any] = json.loads(json.dumps(state))
        committed = int(state["generation"])
        journaled = self._last_journal_generation()
        if journaled > committed:
            aborted = new.setdefault("aborted_generations", [])
            aborted.extend(g for g in range(committed + 1, journaled + 1) if g not in aborted)
        new["generation"] = max(committed, journaled) + 1
        return new

    @staticmethod
    def check_invariants(state: dict[str, Any]) -> None:
        """Exactly the ``active`` pointer's version is in state ACTIVE (none if no pointer)."""
        active = sorted(v for v, e in state["versions"].items() if e["state"] == State.ACTIVE)
        expected = [state["active"]] if state.get("active") else []
        if active != expected:
            raise RegistryError(
                "invariant_violation",
                f"versions in state active {active} != active pointer {expected}",
            )
        kg = state.get("known_good")
        if kg is not None and state["versions"].get(kg, {}).get("state") not in (
            State.ACTIVE,
            State.RETIRED,
        ):
            raise RegistryError("invariant_violation", f"known_good {kg} is not active/retired")

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
        d = self.models_dir / version
        if d.is_symlink() or (d.exists() and d.resolve().parent != self.models_dir.resolve()):
            raise RegistryError("artifact_path", f"{d} is a symlink or escapes the registry")
        return d

    def known_bad_shas(self, state: dict[str, Any]) -> dict[str, str]:
        """model_sha256 -> version, for every rejected or rolled-back version."""
        out: dict[str, str] = {}
        for v, e in state["versions"].items():
            if e["state"] in (State.REJECTED, State.ROLLED_BACK):
                try:
                    out[read_manifest(self.models_dir / v).model_sha256] = v
                except ArtifactError:
                    continue
        return out

    def state_of(self, version: str) -> State | None:
        v = self.read_state()["versions"].get(version)
        return State(v["state"]) if v else None

    def inspect(self) -> dict[str, Any]:
        state = self.read_state()
        aborted = set(state.get("aborted_generations", []))
        uncommitted = [
            e
            for e in self.events()
            if int(e.get("generation", 0)) > int(state["generation"])
            or e.get("generation") in aborted
        ]
        return {**state, "uncommitted_events": uncommitted}

    # -- mutations ------------------------------------------------------------------

    def register(self, src_dir: Path, *, correlation_id: str | None = None) -> str:
        manifest = read_manifest(src_dir)
        version = manifest.model_version
        dest = self.artifact_dir(version)
        with self.locked():
            state = self.read_state()
            if version in state["versions"]:
                raise RegistryError("duplicate_version", f"{version} already registered")
            bad = self.known_bad_shas(state).get(manifest.model_sha256)
            if bad is not None:
                raise RegistryError(
                    "known_bad_artifact",
                    f"model bytes identical to {bad}, which was rejected or rolled back",
                )
            orphan = None
            if dest.exists():  # left by a register that crashed before its commit
                orphan = self.models_dir / f".orphan-{version}-{uuid.uuid4().hex[:8]}"
                os.replace(dest, orphan)
            tmp = Path(tempfile.mkdtemp(dir=self.models_dir, prefix=f".{version}."))
            try:
                for name in ("model.skops", "manifest.json"):
                    if (src_dir / name).exists():
                        shutil.copy2(src_dir / name, tmp / name)
                os.replace(tmp, dest)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            crashpoint("register:after_copy")
            self._apply(
                state,
                version,
                None,
                State.REGISTERED,
                action="register",
                correlation_id=correlation_id,
                orphan_quarantined=orphan.name if orphan else None,
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
        new = self._next_state(state)
        new["versions"][version] = {"state": str(to), "updated_at": _now(), "action": action}
        for k, v in (pointer_updates or {}).items():
            new[k] = v
        self.check_invariants(new)
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
        crashpoint(f"{action}:after_event")
        self._commit(new)
        crashpoint(f"{action}:after_commit")
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
            new = self._next_state(state)
            transitions = []
            for version, to in changes:
                entry = state["versions"].get(version)
                if entry is None:
                    raise RegistryError("unknown_version", version)
                frm = State(entry["state"])
                # Same-state is allowed only for ACTIVE (probation_passed). Anything else
                # (e.g. retired -> retired) means memory and disk disagree: refuse.
                if not (frm == to == State.ACTIVE) and to not in TRANSITIONS[frm]:
                    raise RegistryError(
                        "invalid_transition", f"{version}: {frm} -> {to} not allowed"
                    )
                new["versions"][version] = {
                    "state": str(to),
                    "updated_at": _now(),
                    "action": action,
                }
                transitions.append(
                    {"model_version": version, "from_state": str(frm), "to_state": str(to)}
                )
            new.update(pointer_updates)
            self.check_invariants(new)
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
            crashpoint(f"{action}:after_event")
            self._commit(new)
            crashpoint(f"{action}:after_commit")
            return {"state": new, "event": event}


def ensure_version_dir(registry: Registry, version: str) -> Path:
    d = registry.artifact_dir(version)
    if not d.is_dir():
        raise ArtifactError("model_missing", f"no artifact directory for {version}")
    return d
