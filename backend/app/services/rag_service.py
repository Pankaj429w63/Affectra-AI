import os
from backend.app.schemas.rag import RAGRequest, RAGResponse, RAGContextItem
from backend.app.llm.explanation import get_llm_provider
from backend.app.llm.rag_prompts import RAG_PROMPT_TEMPLATE
from rag.retriever import AffectraRetriever

class RAGService:
    def __init__(self):
        self.provider = get_llm_provider()
        self.retriever = AffectraRetriever()

    def load(self):
        """Pre-loads the retriever (embeddings and FAISS index) during startup."""
        if not self.retriever.is_loaded:
            self.retriever.load()

    def generate_answer(self, request: RAGRequest) -> RAGResponse:
        # 1. Validate query (Pydantic already handles basic validation, but retriever does more)
        # 2. Retrieve top-K relevant chunks
        try:
            results = self.retriever.retrieve(request.question, top_k=request.top_k)
        except Exception as e:
            raise RuntimeError(f"Retrieval failed: {str(e)}")

        # 3. Construct grounded context
        context_items = []
        rag_context_str = ""
        
        if not results:
            rag_context_str = "No relevant context found in the knowledge base."
        else:
            for i, res in enumerate(results):
                context_items.append(
                    RAGContextItem(
                        source=res.source,
                        chunk_index=res.chunk_index,
                        score=res.score,
                        text=res.text
                    )
                )
                rag_context_str += f"[Source: {res.source}, Chunk: {res.chunk_index}]\n{res.text}\n\n"

        # 4. Send the context + question to the existing LLM provider
        prompt = RAG_PROMPT_TEMPLATE.format(
            question=request.question,
            rag_context=rag_context_str.strip()
        )

        try:
            answer_text = self.provider.generate(prompt)
        except Exception as e:
            answer_text = f"An error occurred while generating the RAG answer: {str(e)}"

        # 5. Return the generated answer plus retrieval metadata
        return RAGResponse(
            question=request.question,
            answer=answer_text,
            retrieved_context=context_items
        )

rag_service = RAGService()
