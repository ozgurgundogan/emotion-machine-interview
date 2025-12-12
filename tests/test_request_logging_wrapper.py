import logging
import unittest

from src.request_logging_wrapper import wrap_client_with_request_logging
from src.reranker import RerankResult


class ListHandler(logging.Handler):
    def __init__(self, dest):
        super().__init__()
        self.dest = dest

    def emit(self, record):
        self.dest.append(record)


class DummyIndexer:
    def __init__(self):
        self.calls = []

    def search(self, query, *args, **kwargs):
        self.calls.append(query)
        return [{"tool_id": "a"}, {"tool_id": "b"}]


class DummyReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_n=5, *args, **kwargs):
        self.calls.append((query, [c["tool_id"] for c in candidates], top_n))
        return RerankResult(candidates=candidates[:top_n], notes="dummy")


class DummyPlanner:
    def __init__(self):
        self.calls = []

    def plan(self, query, candidates, max_candidates=10, *args, **kwargs):
        self.calls.append((query, [c["tool_id"] for c in candidates], max_candidates))
        steps = [{"tool_id": candidates[0]["tool_id"], "arguments": {}}] if candidates else []
        return {"strategy": "dummy", "steps": steps}


class DummyClient:
    def __init__(self):
        self.indexer = DummyIndexer()
        self.reranker = DummyReranker()
        self.planner = DummyPlanner()

    def plan_query(self, query, *args, **kwargs):
        cands = self.indexer.search(query)
        reranked = self.reranker.rerank(query, cands, top_n=kwargs.get("count", 5))
        plan = self.planner.plan(query, reranked.candidates, max_candidates=kwargs.get("count", 5))
        return {"query": query, "plan": plan, "candidates": reranked.candidates}


class TestRequestLoggingWrapper(unittest.TestCase):
    def test_wrapper_logs_and_restores_client(self):
        client = DummyClient()
        original_search = client.indexer.search
        original_rerank = client.reranker.rerank
        original_plan = client.planner.plan

        records = []
        logger = logging.getLogger("request_logger_test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        logger.addHandler(ListHandler(records))

        wrapped = wrap_client_with_request_logging(client, logger=logger)
        result = wrapped.plan_query("find tool", request_id="req-1")

        self.assertEqual(result["request_id"], "req-1")
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].request_id, "req-1")
        self.assertEqual(records[0].candidates, ["a", "b"])
        self.assertEqual(records[1].ordered_candidates, ["a", "b"])
        self.assertEqual(records[-1].steps, ["a"])
        self.assertEqual(client.indexer.search.__name__, original_search.__name__)
        self.assertEqual(client.reranker.rerank.__name__, original_rerank.__name__)
        self.assertEqual(client.planner.plan.__name__, original_plan.__name__)


if __name__ == "__main__":
    unittest.main()
