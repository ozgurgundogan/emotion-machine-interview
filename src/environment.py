DATASET_PATHS = [
    "../data/gorilla_openfunctions_v1_train.json",
]

INDEX_PATH = "../index/faiss.index"
METADATA_PATH = "../index/metadata.json"
MISMATCH_PATH = "../evaluation/mismatches.jsonl"

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "gpt-4o-mini"
DEFAULT_PLANNER_MODEL = "gpt-4o-mini"


INDEX_DB_RETRIEVAL_COUNT = 10
APPLY_STD = True
STD_COEF = 0.5
RESPONSE_RETRIEVAL_COUNT = 5
RERANK_RETRIEVAL_COUNT = 5