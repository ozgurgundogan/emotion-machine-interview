import os
from pathlib import Path

import pytest

from src.indexer import Indexer
from src.environment import INDEX_PATH, METADATA_PATH


@pytest.mark.skipif(
    not os.getenv("RUN_INDEX_DB_TEST"),
    reason="Integration test requires prebuilt FAISS index and local model; set RUN_INDEX_DB_TEST=1 to enable.",
)
def test_quick_search():
    if not Path(INDEX_PATH).exists() or not Path(METADATA_PATH).exists():
        pytest.skip("Index/metadata not present; generate index before running integration test.")

    indexer = Indexer(INDEX_PATH, METADATA_PATH)
    indexer.load()  # load FAISS + metadata

    results = indexer.search("Book a flight from Los Angeles to New York for two people on June 15th.")
    assert results  # expect at least one match when index is available

