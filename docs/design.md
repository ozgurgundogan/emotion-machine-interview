Design rationale

Problem we’re solving
- Let a user ask for a task and quickly surface the best-fitting tool(s) from a catalog, without shoving the whole catalog into an LLM prompt. Keep it fast and deterministic by default, with an LLM refinement path when available.

Key choices
- Retrieval-first: pre-embed tools and use FAISS to fetch the top candidates; this keeps recall high and latency low.
- Optional LLM rerank: reorders the retrieved set when an API key is present; otherwise falls back to identity.
- Lightweight planning: deterministic top-1 planner produces a runnable JSON plan; an LLM planner is available for richer sequencing.
- Query segmentation: deterministic delimiter-based by default, or LLM-based when enabled, to split multi-intent requests before retrieval/plan.
- Score gating: optional standard-deviation cutoff removes low-similarity hits before rerank/plan to reduce noise and hallucinated picks.
- Minimal execution stub: the executor records status but actual tool handlers are pluggable, keeping the core pipeline decoupled.
- Streaming-friendly: the backend can stream the plan payload so the UI can show candidates then steps as they arrive.
- Parameter-aware text: descriptions and parameters are normalized before embedding to improve semantic match quality.

Resulting behavior
- Cold-start works without any LLM access (embedding model is local, rerank/planner have deterministic fallbacks).
- Adding LLM credentials only affects rerank/planning; the API surface stays the same.
- Observability is clear: candidates (with scores) and the resulting plan are returned to the frontend for inspection.
- Instrumentation and eval are built-in: enable request-level logging via `request_logging_wrapper`, and run the recall harness to capture mismatches for offline review.
