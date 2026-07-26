import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def build_indices():
    chunks_path = "data/processed/final_chunks.json"
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    texts = [chunk["page_content"] for chunk in chunks]

    # ==========================================
    # PART A: THE DENSE VECTOR INDEX (FAISS)
    # ==========================================
    print("Downloading/Loading the Transformer model (BAAI/bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print("Generating dense vector embeddings... (this takes a few seconds)")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    dimension = embeddings.shape[1] # For this model, it's 384
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    # ==========================================
    # PART B: THE SPARSE KEYWORD INDEX (BM25)
    # ==========================================
    print("Building BM25 keyword index...")
    tokenized_texts = [text.lower().split(" ") for text in texts]
    bm25_index = BM25Okapi(tokenized_texts)

    # ==========================================
    # PART C: SAVE EVERYTHING TO DISK
    # ==========================================
    output_dir = "data/indices"
    os.makedirs(output_dir, exist_ok=True)
    
    faiss_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(faiss_index, faiss_path)
    
    bm25_path = os.path.join(output_dir, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_index, f)
        
    payload_path = os.path.join(output_dir, "payload_metadata.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nIndexing Complete!")
    print(f"FAISS saved to: {faiss_path}")
    print(f"BM25 saved to: {bm25_path}")
    print(f"Payloads saved to: {payload_path}")

if __name__ == "__main__":
    build_indices()