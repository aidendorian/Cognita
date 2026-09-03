from sentence_transformers import CrossEncoder
from config.env import enable_reranker, reranker_top_k
class Reranker:
    def __init__(self):
        self.enabled = enable_reranker
        if self.enabled:
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.top_k = reranker_top_k

    def rerank(self, query: str, documents: list):
        if not self.enabled or not documents:
            return documents

        pairs = [[query, doc["chunk_text"]] for doc in documents]
        scores = self.model.predict(pairs)
        
        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:self.top_k]]