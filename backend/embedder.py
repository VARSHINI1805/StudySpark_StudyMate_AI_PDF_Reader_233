from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load the specified sentence-transformers model
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        # Returns unit-normalized embeddings (better for cosine similarity/L2 on normalized vectors)
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
