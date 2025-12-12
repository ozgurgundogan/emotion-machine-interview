import logging
import unittest
from unittest.mock import patch

from src import planner


class TestPlanner(unittest.TestCase):
    def test_deterministic_plan_picks_first_candidate(self):
        dummy_logger = logging.getLogger("test_planner")
        with patch("src.planner.get_logger", return_value=dummy_logger):
            pl = planner.Planner()
        candidates = [
            {
                "tool_id": "t1",
                "name": "Tool One",
                "api_name": "one.do",
                "parameters": {"required": [{"name": "arg1"}, {"name": "arg2"}]},
            },
            {"tool_id": "t2", "name": "Tool Two", "api_name": "two.do", "parameters": {"required": []}},
        ]
        plan = pl.plan("do something", candidates, max_candidates=2)
        self.assertEqual(plan["steps"][0]["tool_id"], "t1")
        self.assertEqual(set(plan["steps"][0]["arguments"].keys()), {"arg1", "arg2"})
        self.assertEqual(plan["candidates_considered"], ["t1", "t2"])


if __name__ == "__main__":
    unittest.main()
