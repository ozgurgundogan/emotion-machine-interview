import re


class DeterministicSegmenter:

    def __init__(self, delimiters=None, max_segments=3):
        self.delimiters = delimiters or [
            r"\band\b",
            r";",
            r"\.",
            r", then ",
            r" then ",
            r"\bafter\b",
            r"\bnext\b",
            r"\bfinally\b",
            r"&",
            r"\bsubsequently\b",
        ]
        self.max_segments = max_segments

    def segment(self, query):
        if not query:
            return []
        pattern = "|".join(self.delimiters)
        parts = re.split(pattern, query, flags=re.IGNORECASE)
        segments = [p.strip() for p in parts if p.strip()]
        return segments[: self.max_segments]


class LLMBasedSegmenter:

    def __init__(self, model=None, client=None, prompt_builder=None):
        self.model = model
        self.client = client
        self.prompt_builder = prompt_builder or self._default_prompt

    def _default_prompt(self, query):
        return (
            "Split the user request into minimal sub-tasks (2-3 segments max). "
            "Return JSON: {\"segments\": [\"segment1\", \"segment2\", ...]}.\n"
            f"User request:\n{query}"
        )

    def segment(self, query):
        if not self.client:
            raise RuntimeError("LLM client not configured for segmenter.")
        prompt = self.prompt_builder(query)
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.0,
            max_output_tokens=200,
            response_format={"type": "json_object"},
        )
        try:
            data = resp.output_json  # if available
        except Exception:
            import json

            data = json.loads(resp.output_text)
        return data.get("segments") or []
