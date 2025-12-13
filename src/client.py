import json
import os
import time
from pprint import pprint

from src.context_segmenter import DeterministicSegmenter, LLMBasedSegmenter
from src.embedder import Embedder
from src.environment import INDEX_PATH, METADATA_PATH
from src.executor import Executor
from src.indexer import Indexer
from src.logger import get_logger
from src.planner import LLMPlanner, Planner
from src.reranker import OpenAILLMReranker, Reranker

use_llm_planner = os.getenv("USE_LLM_PLANNER", "false").lower() in ("1", "true", "yes")
use_llm_rerank = os.getenv("USE_LLM_RERANK", "false").lower() in ("1", "true", "yes")
use_llm_context_segmenter = os.getenv("USE_LLM_CONTEXT_SEGMENTER", "false").lower() in ("1", "true", "yes")

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
        self.context_segments = LLMBasedSegmenter() if use_llm_context_segmenter else DeterministicSegmenter()
        self.embedder = Embedder()
        self.reranker = OpenAILLMReranker() if use_llm_rerank else Reranker()
        self.planner = LLMPlanner() if use_llm_planner else Planner()


    def plan_query_with_timing(
        self,
        query,
        count: int = 5
    ):
        t0 = time.perf_counter()
        segmented_queries = self.context_segments.segment(query)
        t1 = time.perf_counter()

        candidates = [
            item
            for seg_query in segmented_queries
            for item in self.indexer.search(seg_query)
        ]
        t2 = time.perf_counter()

        rerank_result = self.reranker.rerank(query, candidates, top_n=count)
        t3 = time.perf_counter()

        plan = self.planner.plan(query, rerank_result.candidates, max_candidates=count)
        t4 = time.perf_counter()

        timings_ms = {
            "segment_in_llm_ms": (t1 - t0) * 1000,
            "search_in_vector_db_ms": (t2 - t1) * 1000,
            "rerank_in_llm_ms": (t3 - t2) * 1000,
            "plan_in_llm_ms": (t4 - t3) * 1000,
            "total_ms": (t4 - t0) * 1000,
        }

        pprint(timings_ms)

        return {
            "query": query,
            "plan": plan,
            "candidates": candidates,
            "timings_ms": timings_ms,
        }

    def plan_query(
        self,
        query,
        count: int = 5
    ):
        segmented_queries = self.context_segments.segment(query)

        candidates = [item for seg_query in segmented_queries for item in self.indexer.search(seg_query)]
        rerank_result = self.reranker.rerank(query, candidates, top_n=count)
        plan = self.planner.plan(query, rerank_result.candidates, max_candidates=count)
        return {"query": query, "plan": plan, "candidates": candidates}

    def run_and_print(self, query, count=5):
        result = self.plan_query(query, count=count)
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
    client.run_and_print(user_query, count=5)
