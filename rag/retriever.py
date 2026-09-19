import os
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from rag.ingestion.embeddings import EmbeddingModel
from rag.ingestion.chunker import Chunk
from rag.vectorstore.faiss_store import FAISSStore

@dataclass
class RetrievalResult:
    text: str
    source: str
    chunk_index: int
    score: float

class AffectraRetriever:
    def __init__(self, index_dir: str = "data/rag_vectorstore", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.store = FAISSStore(index_dir=self.index_dir)
        self.embed_model = EmbeddingModel(model_name=embedding_model_name, device="cpu")
        self.is_loaded = False

    def load(self):
        """Loads the FAISS index and metadata from disk."""
        try:
            self.store.load()
            self.is_loaded = True
        except FileNotFoundError:
            raise FileNotFoundError(f"FAISS vector store not found at {self.index_dir}. Please build it first.")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve the top_k most relevant chunks for a given query.
        """
        # Query Validation
        if query is None:
            raise ValueError("Query cannot be None.")
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace-only.")
        
        # Top-K Validation
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")

        if not self.is_loaded:
            self.load()
            
        # Ensure top_k doesn't exceed index size
        actual_top_k = min(top_k, self.store.size())
        if actual_top_k == 0:
            return []

        # Convert query to embedding
        # Reusing embed_chunks which expects a List[Chunk]
        dummy_chunk = Chunk(text=query)
        embeddings_list = self.embed_model.embed_chunks([dummy_chunk])
        
        # Format as 2D numpy array of shape (1, 384)
        query_embedding = np.array(embeddings_list, dtype=np.float32)
        
        # Perform FAISS similarity search
        scores, indices = self.store.index.search(query_embedding, actual_top_k)
        
        results = []
        # scores[0] and indices[0] because query_embedding is a single vector
        for j, faiss_id in enumerate(indices[0]):
            if faiss_id == -1:
                continue # FAISS returns -1 if it doesn't find enough neighbors
                
            if faiss_id < len(self.store.metadata):
                meta_record = self.store.metadata[faiss_id]
                
                res = RetrievalResult(
                    text=meta_record.get("text", ""),
                    source=meta_record.get("source", "unknown"),
                    chunk_index=meta_record.get("chunk_index", -1),
                    score=float(scores[0][j])
                )
                results.append(res)
                
        # FAISS search inherently preserves descending order of similarity score (inner product)
        return results
