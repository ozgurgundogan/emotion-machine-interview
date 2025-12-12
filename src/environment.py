import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for env_file in (ROOT / ".env", ROOT / "local.env"):
    load_dotenv(env_file, override=False)


DATASET_PATHS = [
    p.strip()
    for p in os.getenv("DATASET_PATHS", str(ROOT / "data" / "gorilla_openfunctions_v1_train.json")).split(",")
    if p.strip()
]

INDEX_PATH = os.getenv("INDEX_PATH", str(ROOT / "index" / "faiss.index"))
METADATA_PATH = os.getenv("METADATA_PATH", str(ROOT / "index" / "metadata.json"))
MISMATCH_PATH = os.getenv("MISMATCH_PATH", str(ROOT / "evaluation" / "mismatches.jsonl"))

DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_RERANK_MODEL = os.getenv("DEFAULT_RERANK_MODEL", "gpt-4o")
DEFAULT_PLANNER_MODEL = os.getenv("DEFAULT_PLANNER_MODEL", "gpt-4o")
DEFAULT_SEGMENTER_MODEL = os.getenv("DEFAULT_SEGMENTER_MODEL", "gpt-4o-mini")

INDEX_DB_RETRIEVAL_COUNT = int(os.getenv("INDEX_DB_RETRIEVAL_COUNT", "10"))
APPLY_STD = os.getenv("APPLY_STD", "true").lower() in ("1", "true", "yes", "on")
STD_COEF = float(os.getenv("STD_COEF", "0.5"))
RESPONSE_RETRIEVAL_COUNT = int(os.getenv("RESPONSE_RETRIEVAL_COUNT", "5"))
RERANK_RETRIEVAL_COUNT = int(os.getenv("RERANK_RETRIEVAL_COUNT", "5"))
