from dataclasses import dataclass, field
from typing import List
from rag.ingestion.loader import Document

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        chunks = []
        for doc in documents:
            text = doc.text.strip()
            if not text:
                continue

            start = 0
            chunk_index = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                
                if end < len(text):
                    # Try to find a breaking point
                    last_newline = text.rfind('\n', start, end)
                    if last_newline != -1 and last_newline > start:
                        end = last_newline + 1
                    else:
                        last_space = text.rfind(' ', start, end)
                        if last_space != -1 and last_space > start:
                            end = last_space + 1
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    metadata = doc.metadata.copy()
                    metadata["chunk_index"] = chunk_index
                    chunks.append(Chunk(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                
                # Advance start for the next chunk
                # We want to overlap backwards from the 'end' point
                # Make sure we actually advance
                next_start = end - self.chunk_overlap
                if next_start <= start:
                    next_start = end # Force advancement to prevent infinite loops
                start = next_start
                
        return chunks
