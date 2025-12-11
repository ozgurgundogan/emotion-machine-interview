import json
import os

from src.embedder import Embedder
from src.environment import INDEX_PATH, METADATA_PATH
from src.executor import Executor
from src.indexer import Indexer
from src.logger import get_logger
from src.planner import LLMPlanner, Planner
from src.reranker import OpenAILLMReranker, Reranker

use_llm_planner = os.getenv("USE_LLM_PLANNER", "false").lower() in ("1", "true", "yes")
use_llm_rerank = os.getenv("USE_LLM_RERANK", "false").lower() in ("1", "true", "yes")

class ToolSelectorClient:
    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        logger_name="client",
    ):
        self.logger = get_logger(logger_name)
        self.indexer = Indexer(index_path=index_path, metadata_path=metadata_path)
        self.indexer.load()
        self.embedder = Embedder()
        self.reranker = OpenAILLMReranker() if use_llm_rerank else Reranker()
        self.planner = LLMPlanner() if use_llm_planner else Planner()


    def plan_query(
        self,
        query,
        k: int = 5
    ):
        candidates = self.indexer.search(query)
        rerank_result = self.reranker.rerank(query, candidates, top_n=k)
        plan = self.planner.plan(query, rerank_result.candidates, max_candidates=k)
        return {"query": query, "plan": plan, "candidates": candidates}

    def run_and_print(self, query, k=5):
        result = self.plan_query(query, k=k)
        self.logger.info("Query: %s", query)
        print("\nTop-k tools for query:")
        print(query)
        print("----")
        for cand in result["candidates"]:
            print("\n--- MATCH ---")
            print(f"{cand.get('name')} :: {cand.get('api_name')}")
            print(cand.get("description"))
            print("required:", (cand.get("parameters", {}) or {}).get("required"))
            print("optional:", (cand.get("parameters", {}) or {}).get("optional"))

        print("\nPlan:")
        print(json.dumps(result["plan"], indent=2, ensure_ascii=False))

        executor = Executor(logger=get_logger("executor"))
        exec_result = executor.run(result["plan"])
        print("\nExecution result (stub):")
        print(json.dumps(exec_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    user_query = "Book a flight from Los Angeles to New York for two people on June 15th."
    client = ToolSelectorClient()
    client.run_and_print(user_query, k=5)
