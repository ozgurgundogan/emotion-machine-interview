import functools
import logging


def wrap_client_with_request_logging(client, logger=None):

    log = logger or logging.getLogger("request_logger")

    # Wrap indexer.search
    original_search = client.indexer.search

    @functools.wraps(original_search)
    def search_with_logging(query, *args, **kwargs):
        rid = kwargs.pop("request_id", None)
        results = original_search(query, *args, **kwargs)
        log.info(
            "retrieval",
            extra={
                "request_id": rid,
                "query": query,
                "candidates": [c.get("tool_id") for c in results],
            },
        )
        return results

    # Wrap reranker.rerank
    original_rerank = client.reranker.rerank

    @functools.wraps(original_rerank)
    def rerank_with_logging(query, candidates, top_n=5, *args, **kwargs):
        rid = kwargs.pop("request_id", None)
        res = original_rerank(query, candidates, top_n=top_n, *args, **kwargs)
        ordered_ids = [c.get("tool_id") for c in res.candidates]
        log.info(
            "rerank",
            extra={
                "request_id": rid,
                "query": query,
                "ordered_candidates": ordered_ids,
                "notes": getattr(res, "notes", ""),
            },
        )
        return res

    # Wrap planner.plan
    original_plan = client.planner.plan

    @functools.wraps(original_plan)
    def plan_with_logging(query, candidates, max_candidates=10, *args, **kwargs):
        rid = kwargs.pop("request_id", None)
        plan = original_plan(query, candidates, max_candidates=max_candidates, *args, **kwargs)
        step_tools = [s.get("tool_id") for s in plan.get("steps", [])]
        log.info(
            "plan",
            extra={
                "request_id": rid,
                "query": query,
                "steps": step_tools,
                "strategy": plan.get("strategy"),
            },
        )
        return plan

    # Wrap plan_query to thread request_id through wrapped components.
    original_plan_query = client.plan_query

    @functools.wraps(original_plan_query)
    def plan_query_with_logging(query, *args, request_id=None, **kwargs):
        # Temporarily swap methods to inject request_id
        client.indexer.search = lambda q, *a, **k: search_with_logging(q, *a, request_id=request_id, **k)
        client.reranker.rerank = (
            lambda q, cands, top_n=5, *a, **k: rerank_with_logging(
                q, cands, top_n=top_n, *a, request_id=request_id, **k
            )
        )
        client.planner.plan = (
            lambda q, cands, max_candidates=10, *a, **k: plan_with_logging(
                q, cands, max_candidates=max_candidates, *a, request_id=request_id, **k
            )
        )
        try:
            result = original_plan_query(query, *args, **kwargs)
            # Final log with plan + candidates for correlation
            log.info(
                "result",
                extra={
                    "request_id": request_id,
                    "candidates": [c.get("tool_id") for c in result.get("candidates", [])],
                    "steps": [s.get("tool_id") for s in result.get("plan", {}).get("steps", [])],
                },
            )
            result["request_id"] = request_id
            return result
        finally:
            # Restore original methods
            client.indexer.search = original_search
            client.reranker.rerank = original_rerank
            client.planner.plan = original_plan

    client.plan_query = plan_query_with_logging
    return client
