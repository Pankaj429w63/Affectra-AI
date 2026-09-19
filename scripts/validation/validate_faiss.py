import os
import sys
import time
import numpy as np
from pathlib import Path

# Ensure project root is in python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from rag.ingestion.loader import DirectoryLoader
from rag.ingestion.chunker import RecursiveCharacterChunker
from rag.ingestion.embeddings import EmbeddingModel
from rag.vectorstore.faiss_store import FAISSStore

def validate_faiss():
    print("\n--- BUILDING VECTOR STORE ---")
    
    docs_dir = os.path.join(project_root, "rag", "documents")
    store_dir = os.path.join(project_root, "data", "rag_vectorstore")
    
    t0 = time.time()
    
    # 1. Load documents
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    
    # 2. Chunk
    chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker.split_documents(documents)
    
    # 3. Generate Embeddings
    print("Loading embedding model...")
    embedder = EmbeddingModel(device="cpu")
    print("Generating embeddings...")
    embeddings_list = embedder.embed_chunks(chunks)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    # 4. Prepare metadata
    metadata = []
    for i, chunk in enumerate(chunks):
        meta_record = {
            "chunk_index": i,
            "source": chunk.metadata.get("source", "unknown"),
            "text": chunk.text
        }
        metadata.append(meta_record)
        
    t1 = time.time()
    print(f"Ingestion time (Load + Chunk + Embed): {t1 - t0:.2f} seconds")
    
    # 5. Build FAISS index
    t2 = time.time()
    store = FAISSStore(index_dir=store_dir)
    store.build(embeddings, metadata)
    t3 = time.time()
    print(f"FAISS Build time: {t3 - t2:.2f} seconds")
    
    # Validate exactly what was requested
    print("\n--- FAISS VALIDATION ---")
    
    docs_ok = len(documents) == 1
    chunks_ok = len(chunks) == 7
    embeds_ok = len(embeddings) == 7
    dim_embed_ok = embeddings.shape[1] == 384
    dim_faiss_ok = store.dimension() == 384
    faiss_count_ok = store.size() == 7
    meta_count_ok = len(store.metadata) == 7
    meta_matches_faiss_ok = len(store.metadata) == store.size()
    
    meta_ordered_ok = True
    for i in range(len(store.metadata)):
        if store.metadata[i]["chunk_index"] != i:
            meta_ordered_ok = False
            break
            
    has_nans = np.isnan(embeddings).sum() > 0
    has_infs = np.isinf(embeddings).sum() > 0
    
    print(f"Knowledge base loads exactly 1 document: {docs_ok}")
    print(f"Exactly 7 chunks produced: {chunks_ok}")
    print(f"Exactly 7 embeddings produced: {embeds_ok}")
    print(f"Embedding dimension is 384: {dim_embed_ok}")
    print(f"FAISS index dimension is 384: {dim_faiss_ok}")
    print(f"FAISS index contains exactly 7 vectors: {faiss_count_ok}")
    print(f"Metadata contains exactly 7 records: {meta_count_ok}")
    print(f"Metadata count equals FAISS vector count: {meta_matches_faiss_ok}")
    print(f"Metadata ordering is deterministic: {meta_ordered_ok}")
    print(f"No NaN values: {not has_nans}")
    print(f"No infinite values: {not has_infs}")
    
    # 6. Save FAISS index
    print("\n--- FAISS PERSISTENCE ---")
    t4 = time.time()
    store.save()
    t5 = time.time()
    print(f"FAISS Save time: {t5 - t4:.2f} seconds")
    print(f"Index can be saved: {os.path.exists(store.index_path)}")
    
    # 7. Load FAISS index
    t6 = time.time()
    new_store = FAISSStore(index_dir=store_dir)
    new_store.load()
    t7 = time.time()
    print(f"FAISS Load time: {t7 - t6:.2f} seconds")
    
    load_ok = new_store.index is not None
    loaded_vector_count_ok = new_store.size() == store.size()
    loaded_dim_ok = new_store.dimension() == store.dimension()
    loaded_meta_ok = len(new_store.metadata) > 0
    loaded_meta_count_ok = len(new_store.metadata) == len(store.metadata)
    
    print(f"Index can be loaded: {load_ok}")
    print(f"Loaded index still contains 7 vectors: {loaded_vector_count_ok}")
    print(f"Loaded dimension matches original: {loaded_dim_ok}")
    print(f"Metadata can be loaded: {loaded_meta_ok}")
    print(f"Loaded metadata still contains 7 records: {loaded_meta_count_ok}")
    
    all_passed = (
        docs_ok and chunks_ok and embeds_ok and dim_embed_ok and dim_faiss_ok and 
        faiss_count_ok and meta_count_ok and meta_matches_faiss_ok and meta_ordered_ok and 
        not has_nans and not has_infs and 
        os.path.exists(store.index_path) and load_ok and loaded_vector_count_ok and loaded_dim_ok and 
        loaded_meta_ok and loaded_meta_count_ok
    )
    
    if all_passed:
        print("\nSTATUS: PASS")
    else:
        print("\nSTATUS: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    validate_faiss()
