Flow (backend)

```
User request
  ↓
Embed query (Embedder)
  ↓
FAISS search over tool vectors (Indexer + metadata)
  ↓
Rerank (identity or LLM)
  ↓
Plan (deterministic top-1 or LLM)
  ↓
Return candidates (with scores) + plan; execution is stubbed
```

What the user sees (frontend)
- Land on the UI, enter a backend URL, and submit a query.
- First response chunk shows the top candidates with scores and parameter hints.
- Then the generated plan (steps with tool id/name/api and placeholder arguments) is displayed.
- Streaming mode yields the same payload over time; non-stream returns JSON in one shot.
