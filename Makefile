.PHONY: help install dev seed envs serve rollout-all clean test lint fmt smoke

PY ?= python
PIP ?= $(PY) -m pip
PORT ?= 8000
ENVS_ROOT ?= ./envs

help:
	@awk 'BEGIN { FS = ":.*##"; printf "Targets:\n" } /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install runtime deps + this package in editable mode
	$(PIP) install -e .

dev:  ## Install runtime + dev deps (pytest, ruff, pypdf)
	$(PIP) install -e . -r requirements-dev.txt

seed:  ## Clone + extract the seeded list of repos
	$(PY) -m kernel_synth.scripts.seed_repos

envs:  ## Materialize one RL env folder per extracted candidate
	$(PY) -m kernel_synth.scripts.build_envs

serve:  ## Run the FastAPI viewer (uvicorn) on PORT (default 8000)
	uvicorn kernel_synth.app:app --reload --port $(PORT)

rollout-all:  ## Run baseline rollouts across every env folder
	$(PY) -m kernel_synth.scripts.rollout_all --envs-root $(ENVS_ROOT)

smoke:  ## Import every load-bearing module (catches accidental breakage)
	$(PY) -c "import kernel_synth, kernel_synth.app, kernel_synth.rl.atif, kernel_synth.rl.env, kernel_synth.rl.rewards, kernel_synth.env_builder, kernel_synth.pipeline; print('OK')"

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	ruff check kernel_synth tests

fmt:  ## Auto-format with ruff
	ruff format kernel_synth tests

clean:  ## Remove __pycache__, .pytest_cache, build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
