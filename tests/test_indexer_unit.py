import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


class DummyEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, text):
        return np.array(self.vectors.get(text, [1.0, 0.0]), dtype=np.float32)


class DummyIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []

    def add(self, vecs):
        self.vectors.extend(list(vecs))

    def search(self, qvec, k):
        scores = []
        ids = []
        q = qvec[0]
        for i, v in enumerate(self.vectors):
            scores.append(float(np.dot(q, v)))
            ids.append(i)
        order = np.argsort(scores)[::-1][:k]
        return np.array([[scores[i] for i in order]], dtype=np.float32), np.array([[ids[i] for i in order]], dtype=np.int64)


_INDEX_STORE = {}


def install_stubs(dummy_vectors):
    sys.modules.pop("src.embedder", None)
    sys.modules.pop("faiss", None)
    embedder_mod = types.ModuleType("src.embedder")
    embedder_mod.Embedder = lambda: DummyEmbedder(dummy_vectors)
    sys.modules["src.embedder"] = embedder_mod

    faiss_mod = types.ModuleType("faiss")
    faiss_mod.IndexFlatIP = DummyIndex

    def write_index(idx, path):
        _INDEX_STORE[path] = idx
        Path(path).touch()
        return None

    def read_index(path):
        return _INDEX_STORE.get(path, DummyIndex(0))

    faiss_mod.write_index = write_index
    faiss_mod.read_index = read_index
    sys.modules["faiss"] = faiss_mod


class TestIndexer(unittest.TestCase):
    def test_add_build_and_search(self):
        dummy_vectors = {
            "tool_a": [1.0, 0.0],
            "tool_b": [0.5, 0.5],
            "query": [1.0, 0.0],
        }
        install_stubs(dummy_vectors)
        sys.modules.pop("src.indexer", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = f"{tmpdir}/faiss.index"
            meta_path = f"{tmpdir}/metadata.json"
            indexer_module = importlib.import_module("src.indexer")
            Indexer = indexer_module.Indexer
            indexer_module.APPLY_STD = False

            idx = Indexer(index_path=index_path, metadata_path=meta_path)
            idx.embedder = DummyEmbedder(dummy_vectors)

            idx.add("a", "tool_a", {"name": "A"}, 0)
            idx.add("b", "tool_b", {"name": "B"}, 1)
            idx.build_index()

            idx.load()
            idx.embedder = DummyEmbedder(dummy_vectors)
            results = idx.search("query")
            self.assertEqual(results[0]["tool_id"], "a")
            self.assertGreaterEqual(results[0]["score"], results[-1]["score"])

    def test_build_index_handles_empty_vectors(self):
        dummy_vectors = {}
        install_stubs(dummy_vectors)
        sys.modules.pop("src.indexer", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = f"{tmpdir}/faiss.index"
            meta_path = f"{tmpdir}/metadata.json"
            indexer_module = importlib.import_module("src.indexer")
            Indexer = indexer_module.Indexer

            idx = Indexer(index_path=index_path, metadata_path=meta_path, dim=2)
            idx.embedder = DummyEmbedder(dummy_vectors)

            idx.build_index()
            self.assertTrue(Path(index_path).exists())
            self.assertTrue(Path(meta_path).exists())
            self.assertEqual(json.loads(Path(meta_path).read_text()), [])


if __name__ == "__main__":
    unittest.main()
