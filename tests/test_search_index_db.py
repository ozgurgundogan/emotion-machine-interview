from src.indexer import Indexer
from src.environment import INDEX_PATH, METADATA_PATH


def quick_search(query, filter_std=True):
    indexer = Indexer(INDEX_PATH, METADATA_PATH)
    indexer.load()            # load FAISS + metadata

    results = indexer.search(query)

    print(f"\nTop-{len(results)} tools for query: {query!r}\n")

    for r in results:
        print(f"--- MATCH {r['id']} | score={r['score']:.4f} ---")
        print(f"\t{r.get('name')} :: {r.get('api_name')}")
        print(f"\t{r.get('description')}")
        print("\trequired:", r.get("parameters", {}).get("required"))
        print("\toptional:", r.get("parameters", {}).get("optional"))
        print()

quick_search("Book a flight from Los Angeles to New York for two people on June 15th.")


