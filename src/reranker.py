import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from src.environment import DEFAULT_RERANK_MODEL

@dataclass
class RerankResult:
    candidates: List[Dict[str, Any]]
    notes: str = ""


class Reranker:
    def rerank(self, query, candidates, top_n=5):
        return RerankResult(candidates=candidates[:top_n], notes="identity")


class OpenAILLMReranker(Reranker):
    def __init__(self, model=None):
        self.model = model or os.getenv("LLM_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self._client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

    def _build_prompt(self, query, candidates):
        lines = []
        for i, c in enumerate(candidates):
            req = c.get("parameters", {}).get("required") or []
            req_str = ", ".join([f"{p.get('name')}({p.get('type','?')})" for p in req])
            lines.append(
                f"{i}. id:{c.get('tool_id')} name:{c.get('name')} api:{c.get('api_name')} "
                f"desc:{c.get('description','')[:120]} req:[{req_str}]"
            )
        return (
            "You are a tool reranker. Given a user request and a list of candidate tools, "
            "return the best tools ordered from most relevant to least. "
            "Output JSON: {\"ranked_ids\": [\"tool_id1\", \"tool_id2\", ...]}.\n"
            "User request:\n"
            f"{query}\n\n"
            "Candidates:\n" + "\n".join(lines) + "\n"
        )

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return RerankResult(candidates=[], notes="no_candidates")
        if not self._client:
            return super().rerank(query, candidates, top_n)

        prompt = self._build_prompt(query, candidates)
        try:
            resp = self._client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.0,
                max_output_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.output_text)
            ranked_ids = data.get("ranked_ids", [])
            by_id = {c.get("tool_id"): c for c in candidates}
            ordered = [by_id[tid] for tid in ranked_ids if tid in by_id]
            for c in candidates:
                if c.get("tool_id") not in ranked_ids:
                    ordered.append(c)
            return RerankResult(candidates=ordered[:top_n], notes="llm_rerank")
        except Exception as e:
            return RerankResult(candidates=candidates[:top_n], notes=f"rerank_fallback:{e}")
