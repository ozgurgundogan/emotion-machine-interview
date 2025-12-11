Setup and run

Prereqs
- Python 3.10 or 3.11, Poetry installed.
- Docker + Docker Compose if using the containerized path.

Local (Poetry)
- Install deps: `poetry install`
- Build the FAISS index (required): the paths in `src/environment.py` are relative to `src/`, so run from that directory:
  - `cd src && PYTHONPATH=.. poetry run python -m src.indexer`
- Start the backend API (from repo root): `poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
- Optional LLM rerank/planner: `OPENAI_API_KEY=... USE_LLM_RERANK=true USE_LLM_PLANNER=true poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- Smoke test: `curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"query":"book a flight","stream":false}'`
- Frontend (static): from `frontend/`, run `python -m http.server 3000` and open http://localhost:3000 (you can point the UI at a custom backend URL in the header input).

Dockerized
- Build and run backend + frontend: `docker compose up --build backend frontend`
- Backend listens on http://localhost:8000 and will try to build the index at startup if `index/faiss.index` is missing (it relies on the same relative paths and mounted `./data`/`./index`).
- Frontend serves at http://localhost:3000.
- To enable LLM rerank/planner in the container, pass env vars on the `backend` service (e.g., via `environment:` in compose or `docker compose run -e OPENAI_API_KEY=... -e USE_LLM_RERANK=true -e USE_LLM_PLANNER=true backend ...`).
