import os
import sys
import time
import math
import numpy as np

# Ensure project root is in python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from rag.ingestion.loader import DirectoryLoader
from rag.ingestion.chunker import RecursiveCharacterChunker
from rag.ingestion.embeddings import EmbeddingModel

def validate_embeddings():
    docs_dir = os.path.join(project_root, "rag", "documents")
    
    # 1. Load documents
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    
    # 2. Chunk
    chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker.split_documents(documents)
    
    # 3. Load Model (and time it)
    print("Loading embedding model (this may download it initially)...")
    t0 = time.time()
    embedder = EmbeddingModel(device="cpu")
    t1 = time.time()
    model_load_time = t1 - t0
    
    # 4. Generate Embeddings (and time it)
    print("Generating embeddings...")
    t2 = time.time()
    embeddings = embedder.embed_chunks(chunks)
    t3 = time.time()
    embedding_gen_time = t3 - t2
    
    # 5. Validation
    embeddings_np = np.array(embeddings)
    shape = embeddings_np.shape if len(embeddings) > 0 else (0, embedder.dimension)
    
    has_nans = np.isnan(embeddings_np).sum() if len(embeddings) > 0 else 0
    has_infs = np.isinf(embeddings_np).sum() if len(embeddings) > 0 else 0
    
    # Print results as requested
    print("\n--- EMBEDDING VALIDATION ---")
    print(f"Documents:\n{len(documents)}")
    print(f"\nChunks:\n{len(chunks)}")
    print(f"\nEmbedding shape:\n{shape[0]}, {shape[1]}")
    print(f"\nDevice:\n{embedder.device.upper()}")
    print(f"\nNaN values:\n{has_nans}")
    print(f"\nInfinite values:\n{has_infs}")
    
    print("\n--- PERFORMANCE (Local Measurement) ---")
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print(f"Embedding generation time: {embedding_gen_time:.2f} seconds")
    
if __name__ == '__main__':
    validate_embeddings()
