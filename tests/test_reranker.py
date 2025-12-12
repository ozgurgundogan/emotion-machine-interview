import unittest

from src import reranker


class TestReranker(unittest.TestCase):
    def test_identity_reranker_truncates(self):
        r = reranker.Reranker()
        candidates = [{"tool_id": "a"}, {"tool_id": "b"}, {"tool_id": "c"}]
        res = r.rerank("query", candidates, top_n=2)
        self.assertEqual([c["tool_id"] for c in res.candidates], ["a", "b"])
        self.assertEqual(res.notes, "identity")

    def test_llm_reranker_falls_back_without_client(self):
        r = reranker.OpenAILLMReranker(model="dummy")
        r._client = None
        candidates = [{"tool_id": "x"}, {"tool_id": "y"}]
        res = r.rerank("query", candidates, top_n=1)
        self.assertEqual([c["tool_id"] for c in res.candidates], ["x"])
        self.assertIn("identity", res.notes or "identity")


if __name__ == "__main__":
    unittest.main()
