from typing import Any, Callable, Dict, List

from src.logger import get_logger


class ExecutionError(Exception):
    pass


class Executor:
    def __init__(self, registry=None, logger=None):
        self.registry = registry or {}
        self.logger = logger or get_logger("executor")

    def register(self, tool_id, func):
        self.registry[tool_id] = func

    def call_tool(self, step):
        tool_id = step.get("tool_id")
        args = step.get("arguments", {}) or {}
        fn = self.registry.get(tool_id)

        if not fn:
            self.logger.info("Skipped tool call: %s (handler not registered)", tool_id)
            return {
                "tool_id": tool_id,
                "status": "skipped",
                "reason": "handler_not_registered",
                "arguments": args,
            }

        try:
            self.logger.info("Calling tool: %s args=%s", tool_id, args)
            output = fn(**args)
            self.logger.info("Tool success: %s", tool_id)
            return {"tool_id": tool_id, "status": "ok", "output": output, "arguments": args}
        except Exception as e:
            self.logger.exception("Tool error: %s", tool_id)
            return {"tool_id": tool_id, "status": "error", "error": str(e), "arguments": args}

    def run(self, plan):
        results: List[Dict[str, Any]] = []
        for step in plan.get("steps", []):
            results.append(self.call_tool(step))

        return {"plan_strategy": plan.get("strategy"), "query": plan.get("query"), "steps": results}
