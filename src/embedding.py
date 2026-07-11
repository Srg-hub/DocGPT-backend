from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Load once when the module is imported
_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)


class Embedder:
    """
    Generates embeddings using a singleton model.
    """

    def __init__(self):
        self.model = _model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            convert_to_numpy=True,
        )