Setup and run

Prereqs
- Python 3.10 or 3.11, Poetry installed.
- Docker + Docker Compose if using the containerized path.

Environment configuration
- `.env` and `local.env` are auto-loaded; set `OPENAI_API_KEY` to unlock OpenAI-based rerank/planner/segmenter.
- Retrieval/scoring: `INDEX_DB_RETRIEVAL_COUNT` (default 10) controls FAISS search depth; `APPLY_STD`/`STD_COEF` enable the standard-deviation cutoff; `RESPONSE_RETRIEVAL_COUNT` caps how many candidates go back to the client.
- Models and data: `HF_MODEL_PATH` or `SENTENCE_TRANSFORMER_MODEL` override the embedding model; `DATASET_PATHS` points to the tool corpus (default Gorilla train set); `INDEX_PATH`/`METADATA_PATH` set artifact locations.
- Optional LLM knobs: `USE_LLM_RERANK`, `USE_LLM_PLANNER`, and `USE_LLM_CONTEXT_SEGMENTER` toggle the LLM versions of each stage independently.
- Evaluation: `MISMATCH_PATH` controls where recall mismatches are written (defaults under `evaluation/`).

Local (Poetry)
- (Optional) Create and activate a virtualenv:
  - `python -m venv .venv`
  - `source .venv/bin/activate` (macOS/Linux) or `.venv\\Scripts\\activate` (Windows)
- Install deps: `poetry install`
- Build the FAISS index (required): `PYTHONPATH=.. poetry run python -m src.indexer` (run from `src/` or repo root; paths resolve via `src/environment.py`).
- Start the backend API (from repo root): `poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
- Optional LLM rerank/planner: `OPENAI_API_KEY=... USE_LLM_RERANK=true USE_LLM_PLANNER=true poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- Smoke test: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query":"book a flight","stream":false}'`
- Frontend (static): from `frontend/`, run `python -m http.server 3000` and open http://localhost:3000 (you can point the UI at a custom backend URL in the header input).

Dockerized
- Build and run backend + frontend: `docker compose up --build backend frontend`
- Backend listens on http://localhost:8000 and will try to build the index at startup if `index/faiss.index` is missing (it relies on the same relative paths and mounted `./data`/`./index`).
- Frontend serves at http://localhost:3000.
- To enable LLM rerank/planner in the container, pass env vars on the `backend` service (e.g., via `environment:` in compose or `docker compose run -e OPENAI_API_KEY=... -e USE_LLM_RERANK=true -e USE_LLM_PLANNER=true backend ...`).

Evaluation and tests
- Recall harness: `poetry run python -m evaluation.evaluate` runs recall@k on the Gorilla manual test sets and writes mismatches to `evaluation/mismatches_*.jsonl` (override destination with `MISMATCH_PATH`).
- Integration check: `RUN_INDEX_DB_TEST=1 poetry run pytest evaluation/test_search_index_db.py` asserts the index returns at least one hit for a sample query (requires a prebuilt index and embedding model locally available).
