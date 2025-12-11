import os
from sentence_transformers import SentenceTransformer
from src.environment import DEFAULT_EMBED_MODEL

class Embedder:
    def __init__(self, model_path=None):
        chosen = model_path or os.getenv("HF_MODEL_PATH") or os.getenv("SENTENCE_TRANSFORMER_MODEL") or DEFAULT_EMBED_MODEL
        self.model = SentenceTransformer(chosen)

    def embed(self, text):
        return self.model.encode([text])[0]
