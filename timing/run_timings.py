import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client import ToolSelectorClient  # noqa: E402


QUERIES = [
    "Book a flight from Los Angeles to New York for two people on June 15th.",
    (
        "I need to book a round-trip flight from Los Angeles to New York for two people, "
        "departing on June 15th and returning a few days later. Once I arrive in New York, "
        "I'll need to rent a car for the duration of the trip, preferably from the airport, "
        "and I'd also like recommendations for a good coffee shop near my hotel or nearby attractions "
        "where I can work or relax."
    ),
    (
        "I'm planning a trip from Los Angeles to New York for two people, leaving on June 15th. "
        "Please help me find flight options, arrange a rental car once I land in New York, and also "
        "suggest a few well-rated coffee shops near where I'll be staying so I can grab coffee "
        "or work remotely during the trip."
    ),
]

WARMUP_QUERY = "Warm up: quick flight search from LA to NYC on June 15th."


def collect_timings(client, queries):
    results = []
    for q in queries:
        result = client.plan_query_with_timing(q, count=5)
        timings = result.get("timings_ms", {})
        results.append({"query": q, "timings": timings})
    return results


def plot_timings(timing_results, output_path):
    stages = [
        "segment_in_llm_ms",
        "search_in_vector_db_ms",
        "rerank_in_llm_ms",
        "plan_in_llm_ms",
    ]

    query_labels = [f"Q{i+1}" for i in range(len(timing_results))]
    stage_labels = [s.replace("_ms", "") for s in stages]

    # Build matrix: rows = queries, cols = stages
    data = []
    for item in timing_results:
        timings = item.get("timings", {})
        data.append([timings.get(stage, 0) for stage in stages])

    num_stages = len(stages)
    num_queries = len(data)

    bar_width = 0.25
    x = range(num_stages)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(num_queries):
        offsets = [j + (i - num_queries / 2) * bar_width + bar_width / 2 for j in x]
        ax.bar(
            offsets,
            data[i],
            width=bar_width,
            label=query_labels[i],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(stage_labels, rotation=30)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Pipeline Timing Comparison")
    ax.legend(title="Query")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Timing plot saved to {output_path}")



def main():
    try:
        client = ToolSelectorClient()
    except FileNotFoundError as exc:
        print(f"Failed to load index/metadata: {exc}")
        return

    # Warm up model loads and FAISS caches; excluded from reported timings/plot.
    client.plan_query_with_timing(WARMUP_QUERY, count=1)

    timing_results = collect_timings(client, QUERIES)

    for idx, item in enumerate(timing_results, start=1):
        print(f"\nQ{idx}: {item['query']}")
        for k, v in item["timings"].items():
            print(f"  {k}: {v:.2f} ms")

    output_path = Path(__file__).resolve().parent / "timings.png"
    plot_timings(timing_results, output_path)


if __name__ == "__main__":
    main()
