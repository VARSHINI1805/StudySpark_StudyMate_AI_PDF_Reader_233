import numpy as np

# Optional FAISS import with fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class Retriever:
    def __init__(self, embeddings, chunks, pages=None):
        """
        Dense retriever with optional FAISS; carries page numbers alongside chunks.

        embeddings: np.ndarray with shape (num_chunks, dim), dtype float32
        chunks: list[str] corresponding to embeddings order
        pages: Optional[list[int]] page number per chunk (1-indexed)
        """
        self.chunks = chunks
        self.pages = pages if pages is not None else [None] * len(chunks)
        # Ensure correct dtype/contiguity
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = self.embeddings.shape[1]

        if _HAS_FAISS:
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(self.embeddings)
        else:
            # Pre-normalize for cosine similarity fallback
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            self.normed = self.embeddings / norms

    def retrieve(self, query_embedding, top_k=3):
        """
        Return top_k most similar chunks; includes page numbers when available.
        """
        if len(self.chunks) == 0:
            return []
        k = max(1, min(int(top_k), len(self.chunks)))
        q = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if _HAS_FAISS:
            distances, indices = self.index.search(q, k)
            return [
                (self.chunks[i], self.pages[i], float(distances[0][j]))
                for j, i in enumerate(indices[0])
            ]
        else:
            qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims = (self.normed @ qn.T).ravel()
            top_idx = np.argsort(-sims)[:k]
            return [
                (self.chunks[i], self.pages[i], float(1.0 - sims[i]))  # pseudo-distance
                for i in top_idx
            ]
