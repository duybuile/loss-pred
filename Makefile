.PHONY: install dev run test eval \
        docker-build docker-up docker-down docker-logs docker-restart \
        assess health

# ── Local development ─────────────────────────────────────────────────────────

install:        ## Install production dependencies
	uv sync --no-dev

dev:            ## Install all dependencies including dev (pytest, etc.)
	uv sync

run:            ## Start the API with hot-reload (local)
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:           ## Run the test suite
	uv run pytest tests/ --ignore=tests/test_assessment_agent_architecture_diagram.py -v

eval:           ## Run the evaluation harness against eval_set.json
	uv run python evals/eval.py

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:   ## Build the Docker image
	docker compose build

docker-up:      ## Build (if needed) and start the container in the background
	docker compose up -d --build

docker-down:    ## Stop and remove the container (keeps volumes)
	docker compose down

docker-restart: ## Restart the API container
	docker compose restart api

docker-logs:    ## Tail the API container logs
	docker compose logs -f api

# ── API shortcuts ─────────────────────────────────────────────────────────────

health:         ## Check the /health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

assess:         ## POST new_record.json to /assess and pretty-print the response
	curl -s -X POST http://localhost:8000/assess \
		-H 'Content-Type: application/json' \
		-d "{\"record\": $$(cat data/new_record.json)}" \
		| python -m json.tool

# ── Help ──────────────────────────────────────────────────────────────────────

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
