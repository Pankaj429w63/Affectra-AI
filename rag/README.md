# Affectra AI RAG System

## Phase 4.1 & 4.2: Foundation, Loader, and Chunking

This module implements a Retrieval-Augmented Generation (RAG) pipeline to ground Affectra AI explanations with factual, project-specific knowledge. 

*Note: We are currently in Phase 4.2. Embeddings, FAISS, and the Retriever are intentionally NOT implemented yet.*

---

## 1. Document Loader
The **Document Loader** (`rag/ingestion/loader.py`) is responsible for reading source documents from the file system into memory safely.

- **What it does**: Scans the knowledge directory, opens files as UTF-8 text, ignores empty or invalid files, and wraps them in a `Document` dataclass.
- **Supported file types**: Markdown (`.md`) and Text (`.txt`).
- **Where documents are stored**: `rag/documents/`. (e.g., `emotion_knowledge.md`).
- **How paths are resolved**: Uses Python's robust `pathlib` module to ensure it can be run from the repository root or other directories without hardcoded, fragile Windows paths.
- **Metadata preservation**: The loader automatically attaches the source filename (e.g., `{"source": "emotion_knowledge.md"}`) to each loaded document so we know exactly where facts came from.

---

## 2. Chunking
The **Chunker** (`rag/ingestion/chunker.py`) splits large documents into smaller, manageable pieces (chunks).

- **What chunking means**: Breaking a long text into shorter, consecutive segments.
- **Why RAG needs chunks**: AI models have limited context windows and databases (like FAISS) match specific ideas better when the text is short and focused.
- **Chunk Size**: Configurable. Defaults to `800` characters. This ensures a chunk is roughly 1-2 paragraphs.
- **Chunk Overlap**: Configurable. Defaults to `120` characters. 
- **Why overlap is useful**: Overlap ensures that a sentence split across two chunks doesn't lose its context. It acts as "glue" between chunks.
- **Metadata preserved**: The chunker copies the document's metadata (e.g., source filename) and adds a `chunk_index` (e.g., `0`, `1`, `2`) so we retain the original sequence.

---

## 3. Simple Flow Example

```
Document (emotion_knowledge.md)
       ↓
     Loader
       ↓
Loaded Document Object (Text + Source Metadata)
       ↓
     Chunker
       ↓
[Chunk 0] (chars 0-800)
[Chunk 1] (chars 680-1480, overlaps previous)
[Chunk 2] (chars 1360-2160, overlaps previous)
...
```

---

## 4. Beginner Commands for Validation

If you want to verify the system on Windows PowerShell, run these from the repository root:

### Run the Ingestion Unit Tests
This runs 16 automated tests to guarantee the loader and chunker work perfectly on edge cases.
```powershell
$env:PYTHONPATH="."; python tests/test_rag_ingestion.py
```

### Run the Real Knowledge Base Validation
This script explicitly loads and chunks the real `emotion_knowledge.md` document without building a database.
```powershell
python validate_ingestion.py
```

---

## Future Architecture (Not yet implemented)
When fully implemented in later phases, the RAG system will follow this flow:
Knowledge Documents → Document Loader → Chunking → Embeddings → FAISS → Retriever → **FastAPI RAG Endpoint** → **LLM Context Injection** → Explanation.

---

## 5. Embeddings (Phase 4.3)
**What an embedding is:** An embedding is a numerical representation of a piece of text (a vector of decimal numbers).
**Why RAG needs embeddings:** To search for relevant facts, we mathematically compare the question's embedding with the embeddings of all our document chunks to find the closest match.
**The Model:** We use `sentence-transformers/all-MiniLM-L6-v2`. It is very fast, accurate for basic similarity search, and produces 384-dimensional vectors.
**Why CPU is sufficient:** Our knowledge base is currently just a few files. Running on a CPU takes less than a second to encode everything, so we do not require complex GPU setup.
**Normalization:** The embeddings are normalized, meaning their mathematical "length" is scaled to 1. This ensures that cosine similarity search (which FAISS will use) works optimally.
**Loading:** The embedding model is heavy (~90MB). Our `EmbeddingModel` class ensures it is only loaded *once* per process, rather than repeatedly.

**What will happen in Phase 4.4:** We will create a FAISS database to actually store these embeddings and perform searches against them! FAISS, Retriever, and RAG API are intentionally NOT implemented yet.

---

## 6. FAISS Vector Store (Phase 4.4)
**Why FAISS:** FAISS (Facebook AI Similarity Search) is an incredibly fast and efficient library for storing and searching dense vectors. It is the industry standard for RAG vector databases.
**IndexFlatIP:** Because our embeddings are already normalized (length 1), the inner product (IP) is exactly equivalent to cosine similarity. `IndexFlatIP` does an exact search without approximation, which is perfect for our small knowledge base.
**Storage:** 
- The FAISS index is saved at `data/rag_vectorstore/index.faiss`.
- The metadata mapping (which connects vector IDs to actual text) is saved at `data/rag_vectorstore/metadata.json`.
**Retriever and API:** The FAISS index is successfully built and can be saved/loaded. However, the Retriever logic and the FastAPI RAG integration are **NOT** implemented yet (planned for future phases).

### Run FAISS Validation
```powershell
python validate_faiss.py
```

### Run Vector Store Tests
```powershell
$env:PYTHONPATH="."; python -m pytest tests/test_faiss_store.py -v
```

---

## 7. RAG Final Status (Phase 4.7)

The RAG pipeline is now fully implemented and validated end-to-end.

- **Knowledge Base**: Implemented
- **Chunking**: Implemented
- **Embeddings**: Implemented
- **FAISS**: Implemented
- **Retriever**: Implemented
- **RAG Service**: Implemented
- **FastAPI /rag Endpoint**: Implemented
- **LLM-RAG Integration**: Implemented
- **Complete Validation**: Implemented

**Current Validation uses MockLLMProvider.** The real LLM grounding behavior has not been fully validated because testing the Mock provider guarantees isolated local testing without API keys, but the Mock provider relies on deterministic responses.

**Features Not Yet Implemented:**
- Agents
- Frontend
- Deployment

### Commands for Final Validation
To run the complete RAG validation pipeline:
```powershell
$env:PYTHONPATH="."; python validate_complete_rag.py
```

To run all RAG-related unit tests:
```powershell
$env:PYTHONPATH="."; python -m pytest tests/test_rag.py tests/test_rag_api.py tests/test_retriever.py -v
```

To run the backend server and open Swagger UI:
```powershell
uvicorn backend.app.main:app --port 8000
# Then open: http://localhost:8000/docs
```
