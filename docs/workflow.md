Flow (backend)

```
User request
  ↓
Context segmentation → deterministic delimiters or LLM (USE_LLM_CONTEXT_SEGMENTER)
  ↓
Embed each segment → FAISS search per segment (cosine/IP, optional std-dev cutoff)
  ↓
Merge results → optional rerank (identity or OpenAI-based JSON reranker)
  ↓
Plan → deterministic top-1 with placeholder args or LLM planner
  ↓
Return candidates (with scores) + plan; streaming returns the JSON payload line by line
```

- Retrieval depth is set by `INDEX_DB_RETRIEVAL_COUNT`; scores can be filtered with `APPLY_STD`/`STD_COEF` before rerank/plan.
- The backend enforces the client-facing cap via `RESPONSE_RETRIEVAL_COUNT`.

What the user sees (frontend)
- Land on the UI, enter a backend URL, and submit a query.
- First response chunk shows the top candidates with scores and parameter hints.
- Then the generated plan (steps with tool id/name/api and placeholder arguments) is displayed.
- Streaming mode yields the same payload over time; non-stream returns JSON in one shot.
