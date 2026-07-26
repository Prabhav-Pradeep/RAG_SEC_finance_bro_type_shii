import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

class ProductionRetriever:
    def __init__(
        self, 
        indices_dir: str = "data/indices",
        bi_encoder_model: str = "BAAI/bge-small-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Initializing Production Retriever...")
        
        # 1. Load Indexes and Metadata
        self.faiss_index = faiss.read_index(os.path.join(indices_dir, "faiss_index.bin"))
        
        with open(os.path.join(indices_dir, "bm25_index.pkl"), "rb") as f:
            self.bm25_index = pickle.load(f)
            
        with open(os.path.join(indices_dir, "payload_metadata.pkl"), "rb") as f:
            self.payloads = pickle.load(f)
            
        # 2. Load Embedders
        print("Loading Bi-Encoder...")
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        
        print("Loading Cross-Encoder Reranker...")
        self.reranker = CrossEncoder(cross_encoder_model)
        
        print("Retriever ready!")

    def dense_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Performs vector similarity search in FAISS."""
        query_vector = self.bi_encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(self.payloads):
                results.append({"doc_id": idx, "dense_rank": rank + 1, "payload": self.payloads[idx]})
        return results

    def sparse_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Performs keyword search in BM25."""
        tokenized_query = query.lower().split(" ")
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({"doc_id": idx, "sparse_rank": rank + 1, "payload": self.payloads[idx]})
        return results

    def reciprocal_rank_fusion(self, dense_res: list, sparse_res: list, k: int = 60, top_n: int = 20) -> list[dict]:
        """Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        
        # Score Dense Results
        for item in dense_res:
            doc_id = item["doc_id"]
            rank = item["dense_rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            
        # Score Sparse Results
        for item in sparse_res:
            doc_id = item["doc_id"]
            rank = item["sparse_rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            
        # Sort by combined RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        fused_results = []
        for doc_id, score in sorted_docs:
            fused_results.append({
                "doc_id": doc_id,
                "rrf_score": score,
                "payload": self.payloads[doc_id]
            })
            
        return fused_results

    def retrieve(self, query: str, final_top_k: int = 3) -> list[dict]:
        """Full Retrieval Pipeline: Dense + Sparse -> RRF Fusion -> Cross-Encoder Reranking."""
        # Step 1: Candidate retrieval
        dense_hits = self.dense_search(query, top_k=20)
        sparse_hits = self.sparse_search(query, top_k=20)
        
        # Step 2: RRF Fusion
        fused_candidates = self.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=15)
        
        # Step 3: Cross-Encoder Reranking
        pairs = [[query, candidate["payload"]["page_content"]] for candidate in fused_candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach rerank scores and sort
        for i, candidate in enumerate(fused_candidates):
            candidate["rerank_score"] = float(rerank_scores[i])
            
        final_ranked = sorted(fused_candidates, key=lambda x: x["rerank_score"], reverse=True)[:final_top_k]
        return final_ranked

# --- EXECUTION BLOCK (Interactive Test) ---
if __name__ == "__main__":
    retriever = ProductionRetriever()
    
    test_query = "What were the main revenue figures or financial metrics reported?"
    print(f"\nSearching for: '{test_query}'...\n")
    
    results = retriever.retrieve(test_query, final_top_k=3)
    
    for i, res in enumerate(results):
        print(f"=== RESULT {i+1} (Rerank Score: {res['rerank_score']:.4f}) ===")
        print(f"Metadata: {res['payload']['metadata']}")
        print(f"Content:\n{res['payload']['page_content'][:300]}...\n")