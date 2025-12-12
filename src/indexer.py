import json
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm

from src.embedder import Embedder
from src.environment import INDEX_PATH, METADATA_PATH, INDEX_DB_RETRIEVAL_COUNT, APPLY_STD, STD_COEF
from src.utils import load_functions, generate_function_as_text, normalize


class Indexer:
    def __init__(self, index_path, metadata_path, dim=None):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedder = Embedder()

        self.vectors = []
        self.metadata = []
        self.dim = dim

        self.index = None

    def _init_index(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, tool_id, text, api_info, idx):
        vec = self.embedder.embed(text).astype("float32")

        if self.index is None:
            self._init_index(vec.shape[0])

        self.vectors.append(vec)
        self.metadata.append({
            "id": idx,
            "tool_id": tool_id,
            "text": text,
            **api_info,
        })

    def build_index(self):
        if not self.vectors:
            if self.index is None:
                if self.dim is None:
                    dummy_vec = self.embedder.embed("").astype("float32")
                    self._init_index(dummy_vec.shape[0])
                else:
                    self._init_index(self.dim)
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
            print(f"\nNo vectors to index; saved empty index → {self.index_path}")
            print(f"Metadata saved → {self.metadata_path}\n")
            return

        vectors = np.stack(self.vectors).astype("float32")
        vectors = normalize(vectors)

        self.index.add(vectors)

        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        print(f"\nFAISS cosine index saved → {self.index_path}")
        print(f"Metadata saved → {self.metadata_path}\n")

    def build(self, apis):
        print(f"Embedding API functions (n={len(apis)})...")

        for idx, tool_id in enumerate(tqdm(apis)):
            api = apis[tool_id]
            txt = generate_function_as_text(api)
            self.add(tool_id, txt, api, idx)

        print("Building FAISS index...")
        self.build_index()

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        print("FAISS index + metadata loaded.")

    def search(self, query):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        qvec = self.embedder.embed(query).astype("float32")
        qvec = normalize(qvec[None, :])

        scores, ids = self.index.search(qvec, INDEX_DB_RETRIEVAL_COUNT)

        scores = scores[0]
        ids = ids[0]

        if APPLY_STD:
            mean = scores.mean()
            std = scores.std()
            threshold = mean - STD_COEF * std
            mask = scores >= threshold
            scores = scores[mask]
            ids = ids[mask]

        results = []
        for score, idx in zip(scores, ids):
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                **meta
            })

        return results


def main():
    apis = load_functions()

    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    indexer = Indexer(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )
    indexer.build(apis)


if __name__ == "__main__":
    main()
