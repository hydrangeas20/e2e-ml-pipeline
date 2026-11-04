# mlpipe platform: architecture and operations

This document covers the serving platform added on top of the original DVC/MLflow
pipeline: model lifecycle, controlled rollout, automatic rollback, observability,
and the runbook. The original pipeline (`src/`, `dvc.yaml`, `params.yaml`, MLflow
runs in `mlruns/` and `mlflow.db`) is unchanged, apart from one import-order fix in
`src/evaluate.py`.

## 1. Architecture

```
            offline (CLI, DVC)                     lifecycle (registry/)                 online (FastAPI, one process)
 data ──► validate/transform ──► train ──► package ──► register ──► validate ──► promote ──► serve ──► monitor ──► rollback
          mlpipe/features.py   mlpipe/train.py  artifacts.py  registry.py  validation.py  serving/deployment.py  serving/app.py
                                                                                         └── health gate ──► automatic rollback
```

This is a modular monolith: a single Python package with these module boundaries.

| Concern | Module | Notes |
|---|---|---|
| Feature contract | `mlpipe/schema.py` | `gdm-features/1`; sha256 fingerprint of the canonical schema |
| Feature transform | `mlpipe/features.py` | one transform shared by training and serving; deliberate defects for fault injection |
| Artifact packaging, integrity, safe load | `mlpipe/artifacts.py` | skops + type allowlist; sha256; manifest |
| Training (artifact-producing) | `mlpipe/train.py` | deterministic RF; legacy pickle migration |
| Registry, state machine, event journal | `mlpipe/registry.py` | fcntl lock; atomic `state.json`; fsync'd `events.jsonl` |
| Candidate validation | `mlpipe/validation.py` | 8 ordered checks |
| Deployment control | `mlpipe/serving/deployment.py` | promotion, probation gate, rollback, restart recovery |
| Serving API | `mlpipe/serving/app.py` (re-exported by `app/main.py`) | contracts, admission, timeouts, audit, admin |
| Observability | `mlpipe/serving/observability.py` | Prometheus metrics, JSON logs, request IDs |
| Fault injection | `mlpipe/serving/faults.py` | opt-in, deterministic |
| Diagnosis | `mlpipe/diagnose.py` | layered root-cause workflow |
| Developer tooling | `mlpipe/cli.py`, `mlpipe/demo.py`, `Makefile` | |

### Component classification (from the initial audit)

| Component | Class | Action |
|---|---|---|
| DVC stages, `params.yaml`, MLflow tracking | **Retain** | unchanged; runs and DB preserved |
| `src/evaluate.py` | **Harden** | `mlflow` was used before being imported (`NameError` at import); import moved |
| `models/model.pkl` loaded with `pickle.load` on startup | **Replace** | Arbitrary code execution on load, no version and no integrity check. The same bytes (md5 pinned in `dvc.lock`) are migrated once into a skops artifact, `v1-legacy-rf`. |
| `app/main.py` (untyped `dict` payload, `GET /` health) | **Refactor + Extend** | Same entry point (`app.main:app`), same `/` and `/predict`; strict typed contract plus metadata |
| Dockerfile (root, no health check, unpinned) | **Harden** | non-root, HEALTHCHECK, hash-pinned lock, build-time model seed |
| `src/train.py` grid search | **Retain** | `mlpipe train` is the deterministic, artifact-producing path |

## 2. Model lifecycle

**Artifact** = `models/<version>/model.skops` + `manifest.json`. The manifest holds:

- version, sha256, size;
- estimator class and classes;
- decision threshold;
- feature schema, plus its fingerprint;
- framework versions;
- training data sha256 and params;
- holdout metrics;
- per-feature reference stats;
- provenance: git commit and source.

**Safe loading.** Bytes are re-hashed immediately before deserialization. `skops.io.get_untrusted_types` must return only types in `TRUSTED_EXTRA_TYPES` (currently `sklearn.tree._tree.Tree`); anything else is refused. Pickle is never loaded, except once: the legacy migration, and only after md5 **and** sha256 match the pins.

The sha256 is integrity, not authenticity. Someone who can write both files can forge a matching pair. Signing (e.g. Sigstore) is listed under Limitations.

**States and transitions** (enforced in `registry.TRANSITIONS`):

```
registered ──► validated ──► candidate ──► active ──► retired ──► active   (rollback target)
     │             │             │            │
     └► rejected ◄─┴─────────────┘            └► rolled_back
```

- `validated → active` is used only by `bootstrap`, when nothing is active yet.
- `candidate → validated` is used when a promotion is interrupted, and on startup for candidates left by a crashed promotion (`candidate_released`).
- `rejected` and `rolled_back` are terminal. To ship a fix, register a new version. An artifact whose bytes (sha256) equal a rejected or rolled-back version is refused under any name (`known_bad_artifact`).
- **Invariant**, checked before every commit (`Registry.check_invariants`): exactly the version named by `active` is in state `active`, and `known_good` is `active` or `retired`. A write that would break it is refused.
- **Journal generations** are unique. A generation that was journaled but never committed (failure or crash between the two) is recorded in `aborted_generations` and listed by `mlpipe state` under `uncommitted_events`.

**Pointers** in `state.json`:

- `active`: the version serving traffic.
- `known_good`: the last version that completed probation.
- `probation`: set while a promoted version is under observation.
- `generation`: incremented by every committed change.

## 3. Promotion and rollback

The design is a **single-slot atomic reference swap with a probation window**. It is not a canary: there is no traffic splitting, and none is claimed.

0. **Reconcile.** If an earlier durable write failed after memory changed (see 5), it is retried first. If it still fails, promotion is refused with `state_degraded` (503).
1. **`promote(version, expected_generation?)`**
   - Refused (409) if:
     - another promotion holds the lock;
     - a version is on probation;
     - the generation is stale;
     - the version is not `validated`.
   - A `registered` version is validated first.
2. **`validated → candidate`.** All checks re-run against the bytes on disk, which catches modification after validation.
   - Failure → `rejected`. The active model is untouched.
3. **Commit, then swap.** `candidate → active` and `previous active → retired` are committed in one generation, with `probation` set. Only then does the in-memory reference move. The known-good model stays loaded.
   - If the commit raises but `state.json` shows it landed (e.g. the directory fsync failed after the rename), the activation is completed so that memory matches disk.
   - Bytes identical to a known-bad version are rejected here too (`not_known_bad`).
4. **Probation gate.** It counts only requests served by the probation version, identified by object identity, so late outcomes from the old model are ignored.

   | Rule | Default | Demo |
   |---|---|---|
   | Fail: consecutive failures, any sample count | ≥ 5 | 5 |
   | Fail: error rate, after `min_samples` | > 0.10 after 20 | > 0.2 after 10 |
   | Fail: p95 latency of successes, after `min_samples` | > 250 ms | (default) |
   | Pass: requests observed without failing | `window` = 100 | 200 |

   Failures counted: injected errors, invalid outputs (NaN or out-of-range probabilities), feature-pipeline errors, timeouts and internal errors. Audit-log failures are **excluded**: they are infrastructure, not candidate defects.
5. **Rollback.**
   - Memory is updated first: stop serving the bad model, even if the disk is failing.
   - Then one generation is committed: `bad → rolled_back`, `known_good → active`, `probation = null`.
   - The restored model is smoke-tested; the result is recorded as `recovery_result: verified | smoke_failed`.
   - If that write fails, the service keeps serving known-good, `/readyz` reports `"degraded": true` (still ready), and the write is kept and retried before the next promotion. A restart also heals, because `state.json` still shows probation.
   - Rollback is idempotent: with nothing on probation it is a recorded `noop`.
6. **Restart recovery.**
   - Restart with `probation` still set → restore known-good (`restart_during_probation`).
   - Active fails to load but known-good loads → restore (`active_load_failed_at_startup`).
   - Neither loads → not ready (503) with the reason.
   - Probation set but the known-good fails to load → **not ready**. The candidate is never served without a fallback, and the probation is left on disk for the operator.
   - Leftover `candidate` versions are released to `validated`.
   - The full crash matrix (process killed at each transaction boundary, then restarted) is in [REVIEW.md §3](REVIEW.md#3-crash-recovery-matrix-teststest_crash_recoverypy).

**Quality is not inferred from the gate.** Labels arrive late. The gate measures serving health only. Predictive quality needs labelled outcomes and the diagnosis workflow (§6).

## 4. Serving

- **Contracts.**
  - `POST /v1/predict` takes `{"instances": [...]}` (1–`max_batch_rows`) and returns:
    - `request_id`, `model_version`, `model_sha256`;
    - `schema_version`, `config_version`, `decision_threshold`;
    - `predictions[]`.
  - `POST /predict` (**deprecated**; no removal date set) is the original flat payload. It still returns `prediction` and `confidence` (= max class probability, as before), plus the same metadata.
  - Strictness (`/v1/predict`):
    - integer features must be JSON integers;
    - strings, booleans, `null`, NaN/inf, and unknown or missing fields are rejected;
    - bounds come from the schema.
  - The **only** difference on `/predict`: an integer feature also accepts a finite float with no fractional part, converted before the bounds check (`148.0` → 148). `148.5`, `"148"`, NaN/inf and out-of-range values are rejected on both routes. Full table: [REVIEW.md §5](REVIEW.md#5-api-compatibility).
- **Errors.** `{"error": {"code", "message", "request_id", "details"}}`. Validation details include field location and message, never the submitted values.
- **Admission.**
  - `max_concurrency` inference threads plus `max_queue` waiting; beyond that, an immediate 429.
  - A slot is released when the thread finishes, so capacity stays bounded even after client timeouts.
- **Timeouts.** `request_timeout_s` → 504. *Limitation:* Python threads cannot be interrupted, so the computation runs to completion, holding its slot, and its result is discarded.
  - Discarded results are not audited, not counted as predictions, and not fed to drift monitoring. They are counted in `mlpipe_abandoned_inference_total{outcome}`.
  - If the audit write wins the race against the timeout, the result is returned instead.
  - A hard deadline would need a killable worker process.
- **One server per registry.** The serving process holds an exclusive lock on `<registry>/.serve.lock`. A second server, or a second uvicorn worker, fails at startup.
- **Health.**
  - `GET /livez`: the process is alive.
  - `GET /readyz` (and `/health`): a model is loaded, its artifact on disk still matches its hash, the registry is readable, and the service is not draining.
  - `GET /` is kept unchanged.
- **Shutdown.** On SIGTERM uvicorn stops accepting connections. The lifespan sets `draining` (new requests get 503, readiness goes false) and waits up to `drain_timeout_s` for admitted requests. uvicorn ≥ 0.30 then re-raises the signal, so the exit status is −15 after a clean drain.
- **Audit log.** One JSONL record per returned prediction, holding IDs, version, sha256, config version and output. No feature values.
  - With `audit_required` (the default), a failed write returns 503 and no prediction. The write is all-or-nothing (truncate on partial write).
  - With `audit_fsync=false` (the default), the record is in the page cache, not guaranteed on disk, when the response is sent.
  - Single writer per file.
- **Admin.** `/admin/*` needs `Authorization: Bearer $MLPIPE_ADMIN_TOKEN`. It is disabled (403) when no token is set.

## 5. Observability

| Metric | Labels |
|---|---|
| `mlpipe_requests_total` | route, status |
| `mlpipe_predictions_total` | model_version |
| `mlpipe_prediction_errors_total` | model_version, reason (`injected`, `invalid_output`, `feature_pipeline`, `timeout`, `internal`, `audit`) |
| `mlpipe_request_latency_seconds` (histogram) | route |
| `mlpipe_in_flight_requests` (includes abandoned computations still running) | — |
| `mlpipe_abandoned_inference_total` | outcome (`completed`, `failed`, `cancelled`) |
| `mlpipe_admission_rejected_total` | reason (`overloaded`, `draining`, `not_ready`) |
| `mlpipe_model_load_failures_total` | stage (`startup`, `validate`, `promote`, `bootstrap`) |
| `mlpipe_feature_validation_failures_total` | — |
| `mlpipe_audit_log_failures_total` | — |
| `mlpipe_deployment_events_total` | action, result |
| `mlpipe_rollbacks_total` | trigger |
| `mlpipe_active_model_info` | model_version, model_sha256, schema_version |
| `mlpipe_deployment_generation`, `mlpipe_probation_active` | — |
| `mlpipe_feature_mean_shift` (proxy) | feature |

Logs are one JSON object per line, carrying `request_id`. Keys such as `features`, `payload` and `authorization` are redacted even if passed in by mistake.

### Runbook

- **Latency spike**
  1. Compare the `mlpipe_request_latency_seconds` p95 by route with `mlpipe_in_flight_requests`.
  2. If in-flight sits at `max_concurrency + max_queue` and `admission_rejected{overloaded}` is rising, the service is saturated: scale out or raise `max_concurrency`. The work is CPU-bound and GIL-limited.
  3. If latency rose right after a promotion (`mlpipe_active_model_info` changed), the gate's p95 rule should have tripped. Check `/admin/deployment` → `probation`.
- **Error spike**
  1. Break down `mlpipe_prediction_errors_total` by `reason` and `model_version`.
  2. A single new `model_version` during probation → expect an automatic rollback (`mlpipe_rollbacks_total`).
  3. `reason="audit"` → check disk and permissions at `MLPIPE_AUDIT_LOG_PATH`.
  4. `reason="feature_pipeline"` → see the diagnosis below.
- **Model-loading failures**
  1. Look at `mlpipe_model_load_failures_total{stage}`.
  2. Run `mlpipe events` to find the `failure_reason` (integrity, untrusted types, schema, load, numerical).
  3. `mlpipe validate <version>` reproduces it offline.
- **`/readyz` shows `degraded: true`.** A rollback or probation result did not reach `state.json` (disk full, permissions). Serving is correct. Fix the disk, then either promote (reconciles first) or restart (restores known-good).
- **503 `restart during probation … known-good … failed to load`.** Restore `models/<known_good>/` from backup (or re-import it), then restart. Do not force the candidate active: it never completed probation.
- **Startup error `serving lock … held by pid N`.** Another server is already using this registry. Stop it, or point this one at a different `MLPIPE_REGISTRY_DIR`. Never run `--workers > 1`.
- **`mlpipe state` lists `uncommitted_events`.** A write was journaled and then failed or crashed before its commit. These events are history, not state. The `aborted_generations` field lists their generation numbers.
- **Unexpected prediction changes**
  1. Each response and audit record carries `model_version`, `model_sha256` and `config_version`. Check whether they changed (`mlpipe events` shows when).
  2. If they did not, check `mlpipe_feature_mean_shift` and `/admin/deployment` → `proxy_monitoring` for input shift.
  3. Run `mlpipe diagnose` on recent labelled data, once labels exist.

## 6. Model-quality diagnosis

`mlpipe diagnose --version V --data labelled.csv [--simulate-defect glucose_mmol] [--no-labels]` walks the layers in pipeline order. It reports the first unhealthy layer as the likely root cause; downstream layers inherit upstream symptoms.

| Layer | Signal | Needs labels |
|---|---|---|
| data quality | missing columns, non-numeric, out of bounds, missing-as-zero rate vs training | no |
| feature pipeline | served features ≠ reference transform on the same rows; served shift without raw shift | no |
| model quality | ROC-AUC (reference transform) vs AUC recorded in the manifest (drop > 0.05) | **yes** |
| calibration / threshold | ECE > 0.10, or positive rate vs prevalence off by > 0.15 | **yes** |
| serving | non-zero error/timeout counters, when supplied | no |

Without labels, model quality and calibration are reported as `unknown`, never `ok`. The tests show that label-only problems such as concept drift are invisible to the proxy layers (`test_diagnose_model_quality`).

## 7. Configuration

All settings come from `MLPIPE_*` environment variables (`mlpipe/config.py`) and are validated at startup. Invalid values, such as `max_concurrency=0`, `gate_min_samples > gate_window` or a short admin token, stop the process before it serves. `config_version` is a hash of every non-secret setting.

## 8. Data provenance

- `data/raw/data.csv` is tracked by DVC (md5 `739f3c91…`) on the DagsHub remote, and is not reachable without credentials.
- The committed holdout split `data/splits/test.csv` (154 rows) and the legacy model are in git. The legacy model's MLflow metrics are reproduced exactly from them (`test_legacy_import_preserves_logged_mlflow_metrics`).
- The committed `src/train.py` and `src/evaluate.py` do not match the md5 and size that `dvc.lock` pins for them. This predates this PR (since `2a077b8`). The code that produced the pinned model is therefore not in git, and training is not reproducible from this repository even with the raw data. Evaluation of the legacy model is reproducible (see [REVIEW.md §7](REVIEW.md#7-reproducibility-separated)).
- **Original training cannot currently be reproduced.** The DVC-pinned training code and the original data are both unavailable. What is reproducible: serving (the legacy model gives the original predictions), artifacts (pinned conversion; deterministic new training), and evaluation of the legacy model. See [REVIEW.md §7](REVIEW.md#7-reproducibility-separated).
- The demo's `v2-rf` and the test candidates are trained on the committed holdout split (`data/splits/test.csv`). They are **deployment-testing fixtures**, used to exercise validation, promotion and rollback. They are not validated production candidates, and their metrics are not comparable with `v1-legacy-rf`.
- `mlpipe fetch-data` downloads the public copy linked in the README, pinned by sha256 (`bf81d06e…`). It differs from the DVC copy: 153 of the 154 holdout rows appear in it, and its md5 differs. New candidates trained on it are therefore **not** comparable, metric for metric, with the legacy model.

## 9. Limitations

- Single process, single host. fcntl locks and the JSONL journal are not multi-host safe. A second process on the same host is refused by the serving lock. Two hosts sharing a network filesystem are **not** protected, because flock semantics there are unreliable.
- Crash recovery is tested against process crashes (`os._exit` at named points) only, not against power loss or storage that ignores fsync.
- No time-based end to probation. With no traffic, a candidate stays on probation indefinitely and blocks further promotions. A restart in that state rolls it back.
- No canary, shadow or traffic splitting. Probation exposes 100% of traffic to the candidate until the gate decides; the gate bounds the damage, at 5 consecutive failures by default.
- Integrity only, no signatures. The registry directory must be writable only by the deployer.
- Timeouts do not cancel computation (§4).
- The event journal is fsync'd; the audit log is fsync'd only with `MLPIPE_AUDIT_FSYNC=true`.
- Proxy drift monitoring is a mean-shift heuristic over the last 500 rows. It is not a statistical test and does not trigger rollback.
- **Docker:** this build environment cannot reach any container registry, so the image was validated on a locally assembled Ubuntu 24.04 / Python 3.12.3 base (`--build-arg BASE_IMAGE=…`) instead of `python:3.12-slim`. See the PR for details. CI builds with the default base.
