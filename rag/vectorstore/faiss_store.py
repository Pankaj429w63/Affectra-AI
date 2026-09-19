import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class FAISSStore:
    def __init__(self, index_dir: str = "data/rag_vectorstore"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "metadata.json"
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Build a new FAISS IndexFlatIP (Inner Product) for cosine similarity."""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided to build FAISS index.")
            
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
            
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array, got {embeddings.ndim}D.")
            
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values.")
            
        if np.isinf(embeddings).any():
            raise ValueError("Embeddings contain infinite values.")
            
        if len(embeddings) != len(metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata records.")

        dim = embeddings.shape[1]
        
        # We enforce float32 for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # Using IndexFlatIP for cosine similarity (assuming normalized embeddings)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata

    def save(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("Cannot save an empty index.")
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def load(self):
        """Load the index and metadata from disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"FAISS index files not found in {self.index_dir.resolve()}")
        
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def dimension(self) -> int:
        """Return the embedding dimension of the index."""
        if self.index is None:
            return 0
        return self.index.d
