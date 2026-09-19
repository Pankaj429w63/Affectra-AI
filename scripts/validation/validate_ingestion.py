import os
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from rag.ingestion.loader import DirectoryLoader
from rag.ingestion.chunker import RecursiveCharacterChunker

def validate():
    docs_dir = os.path.join(project_root, "rag", "documents")
    print(f"Loading documents from {docs_dir}")
    
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    
    print(f"\n--- LOADER RESULTS ---")
    print(f"Documents loaded: {len(documents)}")
    for doc in documents:
        print(f"- {doc.metadata['source']} ({len(doc.text)} characters)")
    
    chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker.split_documents(documents)
    
    print(f"\n--- CHUNKER RESULTS ---")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Overlap: {chunker.chunk_overlap}")
    
    print("\n--- SAMPLE CHUNKS ---")
    for chunk in chunks[:2]:
        print(f"[{chunk.metadata['source']} - chunk {chunk.metadata['chunk_index']}]")
        print(chunk.text)
        print("-" * 40)
        
    print("\nQuality Check: Deterministic Run")
    chunks_run2 = chunker.split_documents(documents)
    is_deterministic = True
    if len(chunks) != len(chunks_run2):
        is_deterministic = False
    else:
        for c1, c2 in zip(chunks, chunks_run2):
            if c1.text != c2.text or c1.metadata != c2.metadata:
                is_deterministic = False
    print(f"Deterministic output: {is_deterministic}")
    
    # Check for empty chunks
    empty_chunks = [c for c in chunks if not c.text.strip()]
    print(f"Empty chunks: {len(empty_chunks)}")

if __name__ == '__main__':
    validate()
