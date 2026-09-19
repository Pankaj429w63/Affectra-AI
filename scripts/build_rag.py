import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from rag.ingestion.loader import DirectoryLoader
from rag.ingestion.chunker import RecursiveCharacterChunker
from rag.ingestion.embeddings import EmbeddingsModel
from rag.vectorstore.faiss_store import FAISSStore
from backend.app.core.config import settings

def main():
    print("="*50)
    print("Affectra AI - Building RAG Index")
    print("="*50)

    docs_dir = os.path.join(project_root, "rag", "documents")
    index_dir = os.path.join(project_root, settings.RAG_INDEX_DIR)
    
    # 1. Load Documents
    print(f"[1] Loading documents from: {docs_dir}")
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    if not documents:
        print("Error: No documents found. Please add .md or .txt files to rag/documents/")
        sys.exit(1)
    print(f"    Loaded {len(documents)} document(s).")

    # 2. Chunk Documents
    print("\n[2] Chunking documents")
    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.split_documents(documents)
    print(f"    Generated {len(chunks)} chunks.")

    # 3. Generate Embeddings
    print("\n[3] Generating embeddings")
    embed_model = EmbeddingsModel(model_name=settings.RAG_EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_model.encode(texts)
    print(f"    Embeddings shape: {embeddings.shape}")

    # 4. Build and Save FAISS Index
    print("\n[4] Building FAISS Index")
    store = FAISSStore(index_dir)
    store.build(embeddings, chunks)
    store.save()
    print("    Index build complete.")
    print("="*50)

if __name__ == "__main__":
    main()
