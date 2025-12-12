import logging
import unittest
from unittest.mock import patch

from src.executor import Executor


def add(a, b):
    return a + b


class TestExecutor(unittest.TestCase):
    def test_register_and_call(self):
        dummy_logger = logging.getLogger("test_executor")
        with patch("src.executor.get_logger", return_value=dummy_logger):
            ex = Executor(registry={})
        ex.register("add", add)
        step = {"tool_id": "add", "arguments": {"a": 2, "b": 3}}
        res = ex.call_tool(step)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["output"], 5)

    def test_run_plan_collects_results(self):
        dummy_logger = logging.getLogger("test_executor")
        with patch("src.executor.get_logger", return_value=dummy_logger):
            ex = Executor(registry={"add": add})
        plan = {"strategy": "test", "query": "sum", "steps": [{"tool_id": "add", "arguments": {"a": 1, "b": 4}}]}
        res = ex.run(plan)
        self.assertEqual(res["query"], "sum")
        self.assertEqual(res["steps"][0]["output"], 5)


if __name__ == "__main__":
    unittest.main()
