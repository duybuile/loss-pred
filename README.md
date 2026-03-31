# Torch Assessment Agent

A hybrid AI agent that helps insurance reviewers assess incoming records quickly and confidently. Given a new record, the agent predicts loss likelihood, retrieves similar historical cases, and synthesises a plain-English recommendation with explicit confidence and escalation signals.

## Architecture

The agent uses a **hybrid controller pattern** — deterministic code owns all routing, confidence computation, and escalation decisions; the LLM is invoked once, only for prose synthesis.

```
POST /assess
  → ControllerPolicy          (confidence bands, retrieval gating, conflict detection)
      → predict_loss tool     (sklearn pipeline → trained classifier)
      → retrieve_similar_records tool   (ChromaDB, only when confidence is low)
  → LLMAdapter.synthesize()   (single pass, skipped on forced escalation)
  → AssessResponse
```

See the [Notion architecture page](https://www.notion.so/334549ed3cfa8181866cc0ca380ec50a) for visual diagrams of the data flow, confidence computation, and evaluation harness.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- An Anthropic or OpenAI API key
- Docker + Docker Compose (for containerised deployment)

---

## Quick Start

### Local development

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 3. Start the API
make run
# or: uv run uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
# Build and start
make docker-up

# Tail logs
make docker-logs

# Stop
make docker-down
```

> **Note:** `data/` and `app/artifacts/` are bind-mounted from the host (see [Configuration](#configuration)). Both directories must exist locally before starting the container.

---

## Configuration

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (or OpenAI) | API key for the LLM provider |
| `OPENAI_API_KEY` | Yes (or Anthropic) | Alternative LLM provider |

Copy `.env.example` to `.env` and fill in the key.

### Agent config (`conf/agent.toml`)

| Key | Default | Description |
|---|---|---|
| `llm.provider` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `llm.model` | `claude-sonnet-4-6` | Model name |
| `llm.max_tokens` | `2048` | Max tokens for synthesis response |
| `controller.high_confidence_threshold` | `0.30` | Margin from 0.5 to reach high confidence |
| `controller.medium_confidence_threshold` | `0.15` | Margin from 0.5 to reach medium confidence |
| `controller.retrieval_n_results` | `3` | Number of similar records to retrieve |

### General config (`conf/general.toml`)

| Key | Default | Description |
|---|---|---|
| `artifacts.model_path` | `app/artifacts/model.pkl` | Trained classifier artifact |
| `artifacts.pipeline_path` | `app/artifacts/feature_pipeline.pkl` | Feature pipeline artifact |
| `data.records_path` | `data/records.csv` | Historical records for conflict detection |
| `controller.retrieval_distance_threshold` | `0.5` | Cosine distance above which retrieval is treated as poor quality |

---

## API Reference

### `GET /health`

Liveness check.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /assess`

Run the assessment agent on a new record.

**Request:**
```json
{
  "record": {
    "record_id": "NEW_0001",
    "risk_type": "cyber",
    "territory": "EU",
    "industry": "transport",
    "limit": 3438000,
    "premium": 26904,
    "broker": "Meridian Re",
    "prior_claims": 1,
    "years_trading": 4
  }
}
```

**Response (`AssessResponse`):**
```json
{
  "record_id": "NEW_0001",
  "recommendation": "...",
  "risk_assessment": "likely_loss",
  "confidence_level": "high",
  "key_factors": ["prior_claims", "premium_to_limit"],
  "summary": "...",
  "similar_records_summary": null,
  "second_opinion_recommended": false,
  "review_guidance": "...",
  "tools_used": ["predict_loss"],
  "warnings": []
}
```

**Quick test with the provided example record:**
```bash
make assess
# or manually:
curl -s -X POST http://localhost:8000/assess \
  -H 'Content-Type: application/json' \
  -d "{\"record\": $(cat data/new_record.json)}" | python -m json.tool
```

---

## Running Tests

```bash
make test
# or: uv run pytest tests/ --ignore=tests/test_assessment_agent_architecture_diagram.py -v
```

The suite covers feature engineering, model prediction, controller policy, LLM adapter, agent orchestration, schemas, and eval scoring — 51 tests total.

---

## Running the Evaluation Harness

```bash
make eval
# or: uv run python evals/eval.py
```

Runs the agent against `evals/eval_set.json` (20 records with known outcomes) and scores each recommendation across five dimensions using an LLM judge. Prints a summary and writes full results to `evals/eval_results.json`.

---

## Project Structure

```
├── app/
│   ├── feature_engineering/   # Sklearn transformer classes (StringCleaner, FeatureEngineer, etc.)
│   ├── artifacts/             # model.pkl + feature_pipeline.pkl (not in git)
│   ├── agent.py               # Hybrid orchestrator — controller → tools → LLM synthesis
│   ├── controller.py          # ControllerPolicy — confidence, gating, escalation (deterministic)
│   ├── llm.py                 # Provider-aware LLM adapter (Anthropic / OpenAI)
│   ├── main.py                # FastAPI application — HTTP layer only
│   ├── model.py               # Inference: load artifacts, run pipeline → classifier
│   ├── schemas.py             # AssessRequest, AssessResponse (Pydantic v2)
│   ├── tools.py               # run_predict_loss(), run_retrieve_similar_records()
│   └── vectorstore.py         # ChromaDB wrapper — cosine similarity retrieval
│
├── conf/
│   ├── agent.toml             # LLM + agent + controller thresholds
│   └── general.toml           # Artifact paths, data paths, vectorstore settings
│
├── data/                      # Not in git — mount from host or supply separately
│   ├── records.csv            # Historical records with loss outcomes
│   ├── documents/             # Plain-English summaries, one per record
│   └── new_record.json        # Example record to assess
│
├── evals/
│   ├── eval_set.json          # 20 labelled records with ground truth
│   └── eval.py                # LLM-as-judge harness; run with `make eval`
│
├── notebooks/
│   └── modelling.ipynb        # EDA + model training (Part 1)
│
├── scripts/
│   └── reserialize_pipeline.py  # Re-serialize feature_pipeline.pkl after retraining
│
├── tests/                     # pytest suite — 51 tests
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── Instruction.md             # Original take-home exercise brief
```

---

## Re-serializing the Pipeline

If you retrain the model in the notebook, re-run the serialization script to ensure the pickle references the correct module path:

```bash
uv run python scripts/reserialize_pipeline.py
```

This remaps the `__main__` module path from notebook cells to `app.feature_engineering.transformers`.
