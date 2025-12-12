Modules overview

- `src/environment.py`: centralizes dataset/index paths and default model identifiers (embedding, rerank, planner); paths are defined relative to `src/`.
- `src/embedder.py`: wraps the sentence-transformers model selection and provides `embed(text)` for queries and tool docs.
- `src/indexer.py`: ingests tool/function docs via `utils.load_functions`, embeds them, and builds the FAISS index plus metadata JSON (run as a script to generate the corpus).
- `src/utils.py`: helpers for reading datasets (JSON/JSONL), hashing function specs, normalizing parameters, and vector normalization.
- `src/client.py`: high-level pipeline entrypoint; loads index/metadata, runs retrieval, optional rerank, planning, and can execute a stubbed plan.
- `src/reranker.py`: identity/top-k passthrough reranker and an OpenAI LLM-based JSON reranker (selected via env).
- `src/planner.py`: deterministic top-1 planner with placeholder args and an optional OpenAI JSON planner (requires API key).
- `src/context_segmenter.py`: query segmentation strategies (deterministic delimiter-based and an LLM-backed placeholder) for multi-segment requests.
- `src/executor.py`: executes planned steps by looking up registered tool handlers; unregistered tools are marked as skipped.
- `src/logger.py`: configures loggers/handlers with consistent formatting for console/file output.
- `backend/main.py`: FastAPI service exposing `/api/query`; streams or returns the pipeline result for the frontend.
- `frontend/`: static UI that lets you enter a query, point to a backend URL, and view candidates (with scores) and the generated plan.
- `docker-compose.yml`: runs the backend (with an index bootstrap if missing) and a static frontend server.
- `Dockerfile`: Poetry-based backend image used by the compose service.
