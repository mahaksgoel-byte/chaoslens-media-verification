from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(corpus)

    def retrieve(self, query, k=5):
        bm25_scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(bm25_scores)[-20:]

        query_emb = self.embedder.encode([query])[0]
        sims = []

        for idx in top_indices:
            sim = np.dot(query_emb, self.embeddings[idx])
            sims.append((idx, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        filtered = [(i, sim) for i, sim in sims if sim > 0.3]
        return [self.corpus[i] for i, _ in filtered[:k]]
