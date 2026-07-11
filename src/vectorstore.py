# import os
# import pickle
# import faiss
# import numpy as np


# class FaissVectorStore:

#     def __init__(self, persist_dir: str):
#         self.persist_dir = persist_dir
#         os.makedirs(persist_dir, exist_ok=True)

#         self.index = None
#         self.metadata = []

#     def build(self, embeddings: np.ndarray, metadata: list):

#         embeddings = embeddings.astype("float32")

#         dim = embeddings.shape[1]

#         self.index = faiss.IndexFlatL2(dim)

#         self.index.add(embeddings)

#         self.metadata = metadata

#         self.save()

#     def save(self):

#         faiss.write_index(
#             self.index,
#             os.path.join(self.persist_dir, "faiss.index"),
#         )

#         with open(
#             os.path.join(self.persist_dir, "metadata.pkl"),
#             "wb",
#         ) as f:

#             pickle.dump(self.metadata, f)

#     def load(self):

#         self.index = faiss.read_index(
#             os.path.join(self.persist_dir, "faiss.index")
#         )

#         with open(
#             os.path.join(self.persist_dir, "metadata.pkl"),
#             "rb",
#         ) as f:

#             self.metadata = pickle.load(f)

#     def search(self, query_embedding, top_k=5):

#         distances, indices = self.index.search(
#             query_embedding.astype("float32"),
#             top_k,
#         )

#         results = []

#         for idx, score in zip(indices[0], distances[0]):
#             results.append(
#                 {
#                     "score": float(score),
#                     "metadata": self.metadata[idx],
#                 }
#             )

#         return results

import os
import pickle

import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list,
    ):

        embeddings = embeddings.astype("float32")

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]

        # Cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        self.metadata = metadata

        self.save()

    def save(self):

        faiss.write_index(
            self.index,
            os.path.join(
                self.persist_dir,
                "faiss.index",
            ),
        )

        with open(
            os.path.join(
                self.persist_dir,
                "metadata.pkl",
            ),
            "wb",
        ) as f:

            pickle.dump(
                self.metadata,
                f,
            )

    def load(self):

        self.index = faiss.read_index(
            os.path.join(
                self.persist_dir,
                "faiss.index",
            )
        )

        with open(
            os.path.join(
                self.persist_dir,
                "metadata.pkl",
            ),
            "rb",
        ) as f:

            self.metadata = pickle.load(f)

    def search(
        self,
        query_embedding,
        top_k=10,
    ):

        query_embedding = query_embedding.astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        seen = set()

        for idx, score in zip(indices[0], scores[0]):

            if idx == -1:
                continue

            metadata = self.metadata[idx]

            key = (
                metadata["source"]["source"],
                metadata["source"]["page"],
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(
                {
                    "score": float(score),
                    "metadata": metadata,
                }
            )

        # Higher cosine similarity first
        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results[:4]