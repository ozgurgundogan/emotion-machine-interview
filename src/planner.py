import json
import os

from openai import OpenAI

from src.environment import DEFAULT_PLANNER_MODEL
from src.logger import get_logger


class Planner:
    def __init__(self):
        self.name = "deterministic_top1"
        self.logger = get_logger("planner")

    def plan(self, query, candidates, max_candidates=1):
        if not candidates:
            return {"query": query, "strategy": self.name, "steps": [], "notes": "No candidates."}

        steps = []
        for cand in candidates[:1]:
            params = (cand or {}).get("parameters") or {}
            required = params.get("required", []) or []
            args = {}
            for idx, p in enumerate(required):
                name = p.get("name") or f"arg_{idx}"
                args[name] = "<fill>"

            steps.append(
                {
                    "tool_id": cand.get("tool_id"),
                    "name": cand.get("name"),
                    "api_name": cand.get("api_name"),
                    "arguments": args,
                }
            )

        self.logger.info("Planner strategy=%s steps=%s", self.name, len(steps))
        return {
            "query": query,
            "strategy": self.name,
            "notes": "LLM planner not used; deterministic top-1 plan.",
            "candidates_considered": [c.get("tool_id") for c in candidates[:1]],
            "steps": steps,
        }


class LLMPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.name = "llm_planner"
        self.model = os.getenv("LLM_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        self._client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        self.logger = get_logger(self.name)

    def _format_candidates(self, candidates, limit=10):
        lines = []
        for cand in candidates[:limit]:
            req = cand.get("parameters", {}).get("required") or []
            req_str = ", ".join([f"{p.get('name')}({p.get('type','?')})" for p in req])
            lines.append(
                f"- id:{cand.get('tool_id')} name:{cand.get('name')} api:{cand.get('api_name')} "
                f"desc:{cand.get('description','')[:120]} req:[{req_str}]"
            )
        return "\n".join(lines)

    def _build_prompt(self, query, candidates):
        return (
            "You are a tool-calling planner. Given a user request and a list of candidate tools, "
            "produce a JSON plan of tool invocations.\n"
            "Output JSON format:\n"
            '{"strategy":"llm_planner","steps":[{"tool_id":"...","arguments":{"param":"value"}}]}\n'
            'If an argument is unknown, use the string "<fill>". Do not add explanations.\n\n'
            f"User request:\n{query}\n\n"
            "Candidate tools:\n"
            f"{self._format_candidates(candidates)}\n"
        )

    def plan(self, query, candidates, max_candidates=10):
        if not self._client:
            raise RuntimeError("LLM planner missing client or OPENAI_API_KEY not set.")
        prompt = self._build_prompt(query, candidates[:max_candidates])
        resp = self._client.responses.create(model=self.model,
                                             input=prompt,
                                             temperature=0.2,
                                             max_output_tokens=300,
                                             response_format={"type": "json_object"})
        plan = json.loads(resp.output_text)
        plan["query"] = query
        plan["strategy"] = self.name
        plan["candidates_considered"] = [c.get("tool_id") for c in candidates[:max_candidates]]
        return plan
