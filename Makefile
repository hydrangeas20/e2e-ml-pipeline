PY ?= python
REG ?= registry
URL ?= http://127.0.0.1:8000

.PHONY: install lint typecheck test test-fast check demo prepare serve docker-build docker-smoke

install:            ## dev environment from the hash-pinned lock
	$(PY) -m pip install --require-hashes -r requirements/dev.lock
	$(PY) -m pip install --no-deps -e .

lint:
	ruff check mlpipe tests app
	ruff format --check mlpipe tests app

typecheck:
	mypy

test:               ## full suite, including the real-server demo test
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

check: lint typecheck test

prepare:            ## legacy model -> skops artifact, bootstrap as active
	mlpipe import-legacy --registry $(REG)
	mlpipe bootstrap --registry $(REG) v1-legacy-rf

serve:
	MLPIPE_REGISTRY_DIR=$(REG) uvicorn app.main:app --host 127.0.0.1 --port 8000

demo:               ## rollout -> injected failure -> automatic rollback, real server
	mlpipe demo --workdir demo_run

docker-build:
	docker build -t mlpipe:dev .

docker-smoke:       ## build, run, check health + predict + non-root, stop
	./scripts/docker_smoke.sh
