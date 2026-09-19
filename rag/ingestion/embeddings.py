import torch
from typing import List
from sentence_transformers import SentenceTransformer
from rag.ingestion.chunker import Chunk

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # Load the model once
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # Determine the dimension of the embeddings by looking at the model itself
        self.dimension = self.model.get_embedding_dimension()

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        if not chunks:
            return []
            
        texts = [chunk.text for chunk in chunks]
        
        # normalize_embeddings=True helps with cosine similarity
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # return as list of lists (numerical vectors)
        return embeddings.tolist()
