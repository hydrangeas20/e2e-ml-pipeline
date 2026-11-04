# Reliability review of PR #1 (deployment integrity, recovery, concurrency)

**Scope:** the MVP at `a0c9a9c` (branch `platform/mvp-rollout`).

**Test environment:**
- Linux x86-64, 2 vCPU.
- Python 3.12.3 and 3.13.18; dependencies from `requirements/dev.lock`.
- Container registries are blocked by the build environment's egress policy, so the local Docker validation used a stand-in base (see §7).

Line numbers refer to `a0c9a9c` ("was") and to the review head ("now").

## 1. Findings

**Status key:**
- **Confirmed:** a regression test fails at `a0c9a9c` and passes now (`tests/test_review_regressions.py`, R1–R12), or the failure was reproduced directly.
- **Hypothesis:** reasoned from the code; no failing test.

| # | Sev | Status | Finding | Location (was → now) | Fix |
|---|---|---|---|---|---|
| R1 | **High** | Confirmed | **Restart during probation with an unloadable known-good served the unproven candidate with no health gate, and made it the in-memory known-good.** The startup fallback path loaded `active` and set `_known_good = good_lm or lm`. | `deployment.py:281,317` → `:305` | Fail closed: not ready, explicit reason, probation left on disk for the operator. |
| R2 | **High** | Confirmed | **After a rollback whose state write failed, the next promotion went ahead on stale durable state.** `state.json` ended with two versions in `active` (the rolled-back `v2-rf` and the new candidate). The `persist_degraded` flag was informational only; nothing retried or blocked. | `deployment.py:530,614` → `:354,603,677` | The failed write is kept as a pending closure. `promote()` reconciles first and refuses with `state_degraded` (503) while it still fails. Readiness reports `degraded`. |
| R12 | **High** | Confirmed | **The registry accepted writes that break the single-active invariant.** Same-state transitions (`retired → retired`) were allowed, and the pointers were never checked against version states. This is why R2 could happen. | `registry.py:273` → `:190,380` | `check_invariants()` before every commit: exactly the `active` pointer is ACTIVE; `known_good` is ACTIVE or RETIRED. Same-state transitions are allowed only for ACTIVE. |
| R10 | Medium | Confirmed | **A promote commit that reached `state.json` but raised afterwards left disk and memory disagreeing.** Example: a failed directory fsync after the rename. Disk said `v2` was active on probation; memory kept serving `v1`; `promote_aborted` failed silently. | `deployment.py:467` → `:526` | Re-read state after the error. If the commit landed, complete the activation; otherwise abort as before. |
| R11 | Medium | Confirmed (found by a crash test) | **A crash between `select_candidate` and the promote commit left the version permanently in `candidate`.** Promotion requires `validated`, so it could not be retried without hand-editing `state.json`. | `deployment.py:424` (no recovery) → `:292` | On startup, release leftover candidates to `validated` (`candidate_released` event). They were never active, so this is not a reactivation. |
| R3 | Medium | Confirmed | **Journal generations were reused after a failed commit.** The event was fsync'd, the commit failed, and the next write took the same generation number. That made the orphan event look committed, so `inspect` no longer flagged it. The audit trail was misleading. | `registry.py:232` → `:173` | Allocate `max(state, journal tail) + 1`. Skipped generations are recorded as `aborted_generations` and reported as uncommitted. |
| R4 | Medium | Confirmed | **A request that timed out (504) still wrote a prediction audit record and fed the drift monitor.** The abandoned computation finished in the background, so the audit log claimed a prediction the client never got. | `app.py:245-260` → `:143,274` | Per-request ticket: the worker commits (audits) or the handler abandons, under one lock. If the commit wins the race, the audited result is returned. New counter `mlpipe_abandoned_inference_total{outcome}`. |
| R5 | Medium | Confirmed | **Nothing stopped two server processes (or `uvicorn --workers N`) from sharing one registry.** Each would run its own in-memory probation and rollback (split-brain). This was documented, not enforced. | `app.py:157` → `registry.py:108`, `app.py:175` | Exclusive non-blocking `flock` on `<registry>/.serve.lock` held for the process lifetime. A second server fails to start with a clear error. |
| R7 | Medium | Confirmed | **Known-bad model bytes could come back under a new version name.** A rolled-back or rejected artifact, re-registered as `v2-again`, would validate and promote. | `registry.py:169` → `:268`; `deployment.py:485` | `register()` and `promote()` refuse artifacts whose sha256 matches a rejected or rolled-back version (`known_bad_artifact`, `not_known_bad`). |
| R6 | Low | Confirmed | **The schema check trusted the manifest's fingerprint field without recomputing it from `feature_schema`.** A manifest could declare `gdm-features/9` while carrying the v1 fingerprint. | `validation.py:118` → `:118` | The metadata check recomputes and compares. |
| R8 | Low | Confirmed | **A crash after copying an artifact but before the register commit blocked re-registration** (`duplicate_version`). | `registry.py:169` → `:271` | The orphan directory is quarantined to `.orphan-<v>-<id>` and the registration proceeds; the event records it. |
| R9 | Low | Confirmed | **A symlinked `models/<version>` directory was followed**, so artifacts could be read from outside the registry. The model-file symlink check did not cover the directory. | `registry.py:148` → `:221` | `artifact_dir()` refuses symlinked directories or paths that resolve outside `models/`. |
| T1 | Low | Confirmed | **`scripts/docker_smoke.sh` exited 141 (SIGPIPE).** `curl … \| head -c 300` under `pipefail` failed whenever the response exceeded the read. This is why "Docker green" was not repeatable locally. | `docker_smoke.sh:28` → `:29` | Capture into a variable, then truncate. |
| H1 | Low | Hypothesis (by inspection) | **Gate bookkeeping, including a rollback's fsync'd writes, and readiness re-hashing ran on the asyncio event loop**, stalling all requests for the I/O duration. Not measured. | `app.py:332`, readiness → `app.py:388` | `asyncio.to_thread` for `record_outcome` and `/readyz`. |
| P1 | Info | Confirmed, **pre-existing** (not introduced by this PR) | **`dvc.lock` pins md5/size for `src/train.py` (5,047 B) and `src/evaluate.py` (2,551 B) that match neither the committed files (5,201 / 2,703 B) nor their LF-normalised forms.** The code that produced the pinned model is not in git. | `dvc.lock`, `src/` since `2a077b8` | Documented. Training is not reproducible from the repository. |
| P2 | Info | Confirmed, pre-existing | The demo's `v2-rf` and the test candidates are trained on `data/splits/test.csv`, the legacy *holdout*. Their manifest provenance says so, but their metrics are not comparable with `v1-legacy-rf`. | `mlpipe/demo.py`, `tests/conftest.py` | Documented. They are fixtures for exercising deployment mechanics, not model candidates. |

**Verified correct, with no defect found.** At `a0c9a9c`, the 21 new gate and concurrency tests in `tests/test_gate_concurrency.py` already passed. They are characterization tests and now guard against regressions. They cover:
- attribution under concurrent promotion *and* rollback;
- a request spanning a swap;
- late results (success and failure) from a rolled-back candidate not touching the next candidate's gate, and not causing a second rollback;
- simultaneous manual and automatic rollback (exactly one event);
- concurrent promotions sharing one stale counter;
- the gate's statistics.

## 2. Deployment state transitions (as implemented now)

Order of side effects within one operation: verify → journal (fsync) → `state.json` (atomic replace + dir fsync) → memory → metrics/readiness. The exception is rollback (marked †).

| Operation | Durable effect (one generation each) | Memory | Readiness |
|---|---|---|---|
| `register` | `models/<v>/` copied (orphan quarantined) → `registered` | — | — |
| `validate` | `registered → validated \| rejected` (+ failure reason) | — | — |
| `bootstrap` (nothing active) | `validated → active`; `active = known_good = v` | load, set active and known-good | ready |
| `promote` | ① reconcile pending writes (refuse if they still fail) ② `validated → candidate` ③ re-verify bytes, known-bad check (fail → `rejected`) ④ `candidate → active`, `prev → retired`, `probation = {v}` | swap reference after ④; known-good stays loaded | unchanged (ready) |
| probation pass (`window` outcomes) | `known_good = v`, `probation = null` | known-good = v | ready |
| health-gate fail / manual rollback † | **memory first**, then `bad → rolled_back`, `good → active`, `probation = null`, `recovery_result` | active = known-good | ready; `degraded` if the write failed |
| repeated rollback | `noop` metric only | — | — |
| re-promote rejected / rolled-back | refused (`invalid_state`); identical bytes under a new name refused (`known_bad_artifact`) | — | — |
| restart | release stale `candidate`s; if probation was set and known-good loads: rollback (`restart_during_probation`); if the active artifact fails and known-good differs: rollback (`active_load_failed_at_startup`) | load | ready or 503 with reason |

† Rollback stops serving the bad model before any durable write, so a broken disk cannot keep it in service.

## 3. Crash-recovery matrix (`tests/test_crash_recovery.py`)

**Method.** A child process (`tests/crash_harness.py`, or a real uvicorn server) is killed with `os._exit` at a named `crashpoint()`: no `finally` blocks, no atexit. A **fresh process** then restarts from disk.

**Coverage limit.** This covers process crashes only. Data already handed to the kernel survives. Power loss and storage that ignores fsync are **not** simulated.

| Crash point | Serves after restart | Durable state after restart | Ready | Recovery | Can bad candidate return? | Audit trail |
|---|---|---|---|---|---|---|
| before promote commit (`promote:after_event`) | v1 | v2 released `candidate → validated`; promote event marked aborted | yes | automatic; promotion can be retried | n/a (never active) | promote event listed as uncommitted; generations unique |
| after promote commit (`promote:after_commit`, `…before_activate`, `…after_activate`) | v1 | v2 `rolled_back`; one active (v1) | yes | automatic | no (`invalid_state`) | `rollback`, trigger `restart_during_probation` |
| clean exit during probation | v1 | v2 `rolled_back` | yes | automatic | no | as above |
| during automatic rollback, memory swapped, nothing durable (`rollback:after_swap`) | v1 | v2 `rolled_back` | yes | automatic | no | ⚠ committed trigger is `restart_during_probation`; the health-gate reason is lost (the per-request errors remain in metrics/logs) |
| rollback journaled, not committed (`rollback:after_event`) | v1 | v2 `rolled_back` | yes | automatic | no | health-gate rollback event listed as uncommitted, then a committed `restart_during_probation` event |
| after rollback commit | v1 | v2 `rolled_back` | yes | n/a | no | one `health_gate` rollback |
| rollback write fails, process keeps running (R2) | v1 (memory) | lags until reconciled | yes, `degraded: true` | automatic on next promote, or on restart | no | rollback event appended when reconciled |
| real uvicorn killed mid-rollback (`rollback:after_swap`) | v1 | v2 `rolled_back` | yes | automatic | no | no failed v2 request audited (failed requests never write audit records) |
| corrupt active artifact (active = known-good) | **nothing** | unchanged | **no** (503, `integrity: …`) | **manual**: restore the artifact or bootstrap a new registry | — | — |
| known-good missing during probation (R1) | **nothing** | unchanged; probation kept | **no** (503, names the known-good) | **manual** | no | — |
| crash during `register` after copy | — | version absent; directory orphaned | — | automatic on re-register (quarantine) | — | event records `orphan_quarantined` |

## 4. Health-gate behaviour (`tests/test_gate_concurrency.py`)

| Question | Behaviour now |
|---|---|
| Small samples | The error rate is evaluated only once `n ≥ min_samples` (default 20). The comparison is strict: 2/20 = 0.10 passes, 3/21 fails. At n = 20 one error is worth 5 points, so this is a coarse estimator. A Wilson bound would be the next step if false rollbacks appear. |
| Consecutive failures | They trip at any sample count (default 5). This is the only rule active below `min_samples`. |
| p95 | Linear interpolation (`numpy.percentile`) over *successful* requests, only once ≥ `min_samples` successes. With 20 samples, one slow request moves p95 only to 59.5 ms; two put it at the slow value. |
| Low traffic | Passing needs `window` (default 100) outcomes. There is no time-based pass. |
| No traffic | Probation persists indefinitely and further promotions are refused (`probation_in_progress`). A restart in that state rolls back. **Owner decision:** add a maximum probation duration, or accept this. |
| What counts as a candidate failure | injected errors, invalid outputs, feature-pipeline errors, internal errors, **timeouts** |
| What does not count | 422 (client input), 429 (admission), 503 audit failures (infrastructure), late outcomes from a model that is no longer the probation subject |
| Rollback errors | A smoke-test failure of the restored model is recorded as `recovery_result: smoke_failed` (the model is still restored; there is nothing better). A persistence failure is handled as R2. |
| Rejected / rolled-back versions | Terminal states; identical bytes under a new name are refused (R7). |

## 5. API compatibility

**Original behaviour.** I ran the original `app/main.py` from `37995fe` against the pinned pickle.

| Input to `POST /predict` | Original `37995fe` | Now |
|---|---|---|
| valid payload | 200 | 200 (identical prediction and confidence; tested) |
| integer feature as `148.0` | 200 | **422** |
| non-integral `148.5` | 200 | **422** |
| numeric string `"148"` | 200 | **422** |
| out of bounds (`Glucose: 9999`) | 200 | **422** |
| `NaN` | 200 (RF accepts missing values) | **422** |
| `Infinity` | 500 | 422 |
| unknown field | 500 | 422 |
| missing feature | 500 | 422 |
| keys in a different order | **500** (sklearn feature-order check) | **200** |
| overload | unbounded | 429 beyond `max_concurrency + max_queue` |

**Recommendation (owner decision; not implemented):**
- Keep `/v1/predict` strict.
- Do not loosen `/predict` silently.
- If existing clients send integral floats, the lowest-risk option is an explicit, documented compatibility rule on the **legacy route only**: accept integral floats (`148.0` → 148) and nothing else. That change is lossless.
- Accepting strings, NaN or out-of-bounds values would reintroduce silent garbage-in behaviour. Previously those inputs produced predictions with no warning.
- Versioning is already in place (`/v1/predict`). A deprecation date for `/predict` is preferable to a compatibility mode flag, because a flag doubles the contract surface the tests must cover.

## 6. Timeouts and capacity

- **Bound.** Admitted work, including abandoned computations, is capped at `max_concurrency + max_queue`. A slot is released only when the worker thread finishes (`test_timeout_returns_504_and_keeps_slot_until_work_finishes`). `mlpipe_in_flight_requests` counts that work.
- **Metrics.**
  - Client timeouts: `mlpipe_prediction_errors_total{reason="timeout"}`.
  - What happened to the abandoned work: `mlpipe_abandoned_inference_total{outcome=completed|failed|cancelled}`.
  - Neither is counted in `mlpipe_predictions_total` or in the audit log (R4).
- **Shutdown.**
  - SIGTERM → draining: new requests get 503 and readiness goes false.
  - The service waits up to `drain_timeout_s` for admitted work, then queued (unstarted) futures are cancelled.
  - Running threads finish before the interpreter exits. Python threads cannot be interrupted, so shutdown time is bounded by the slowest in-flight inference.
- **Cancellation.** Not feasible with in-process threads. A hard deadline needs a killable worker process; that is out of scope for this architecture.

## 7. Reproducibility, separated

| Kind | Status | Evidence |
|---|---|---|
| Serving | **Reproducible** | `v1-legacy-rf` (skops) gives the same prediction and confidence as the original pickle path (`test_legacy_response_matches_original_implementation`). |
| Artifact | **Reproducible given the pinned pickle** | Legacy conversion unpickles only after md5 (`dvc.lock`) and sha256 pins match. New training is deterministic: two runs give bit-identical probabilities (`test_training_is_deterministic`). |
| Evaluation | **Reproducible** for the legacy model | The committed holdout plus the pinned model give the MLflow-logged training-run metrics exactly (`test_legacy_import_preserves_logged_mlflow_metrics`). The fixed `src/evaluate.py`, run as a script on a scratch copy, reproduces evaluate run `96f2e540` on all six metrics. The original crashes with `NameError`. |
| Training | **Not reproducible** | (a) `data/raw/data.csv` exists only on the DagsHub DVC remote. (b) The committed `src/train.py` is not the version pinned in `dvc.lock` (P1). (c) The public copy differs from the original: in the holdout, the public row at index 3 has `BloodPressure` 66 where the pinned split has 55; the other 153 holdout rows match position for position. The byte size also differs (DVC 23,872 B vs 23,873–23,875 B for CRLF rebuilds of the public file), so there are further differences outside the holdout. The original cannot be reconstructed. |

## 8. Verification record

| Check | Before (`a0c9a9c`) | After |
|---|---|---|
| `make check` (ruff, `mypy --strict`, pytest, Python 3.12) | 81 passed | **126 passed** |
| Review regressions R1–R12 | **12 failed** | 12 passed |
| Gate/concurrency characterization (21) | 21 passed | 21 passed |
| Crash-recovery matrix (12, subprocess) | not runnable (no crash points existed) | 12 passed |
| `make demo` (real uvicorn) | passed (27 assertions) | passed (27 assertions) |
| Docker smoke (local, stand-in base) | **exit 141** (T1) | passed: healthy, uid 10001, 422 on bad input, graceful stop |
| CI on the PR | green at `a0c9a9c` | see the PR for the run on the review head |
