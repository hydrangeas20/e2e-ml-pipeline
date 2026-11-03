# Serving image for the mlpipe platform. Dependencies are installed from a
# hash-pinned lock; the legacy model is migrated to a verified skops artifact at
# build time (from the pickle pinned in dvc.lock) and bootstrapped as active.

# BASE_IMAGE must provide python3.12 + pip on PATH and groupadd/useradd.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements/serve.lock requirements/serve.lock
RUN pip install --require-hashes -r requirements/serve.lock
COPY pyproject.toml ./
COPY mlpipe ./mlpipe
COPY app ./app
RUN pip install --no-deps .

FROM base AS seed
COPY data/splits/test.csv data/splits/test.csv
COPY mlruns/0/models/m-ac8443cec0b64c03929f2d97d344b3c5/artifacts/model.pkl \
     mlruns/0/models/m-ac8443cec0b64c03929f2d97d344b3c5/artifacts/model.pkl
RUN mlpipe import-legacy --registry /seed/registry \
 && mlpipe bootstrap --registry /seed/registry v1-legacy-rf

FROM base AS runtime
RUN groupadd --system --gid 10001 mlpipe \
 && useradd --system --uid 10001 --gid mlpipe --home-dir /nonexistent --shell /usr/sbin/nologin mlpipe \
 && mkdir -p /var/lib/mlpipe \
 && chown mlpipe:mlpipe /var/lib/mlpipe
COPY --from=seed --chown=mlpipe:mlpipe /seed/registry /var/lib/mlpipe/registry
ENV MLPIPE_REGISTRY_DIR=/var/lib/mlpipe/registry \
    MLPIPE_AUDIT_LOG_PATH=/var/lib/mlpipe/predictions.jsonl \
    MLPIPE_ENABLE_FAULT_INJECTION=false
# MLPIPE_ADMIN_TOKEN is intentionally not set: pass it at runtime (env or secret).
USER 10001:10001
VOLUME ["/var/lib/mlpipe"]
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"]
STOPSIGNAL SIGTERM
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "15", "--no-server-header"]
