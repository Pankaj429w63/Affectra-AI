import torch
from typing import List
from rag.ingestion.chunker import Chunk

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def get_embedding_dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        if not chunks:
            return []
            
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
