from backend.app.agents.base import BaseAgent
from backend.app.agents.context import AgentContext
from rag.retriever import AffectraRetriever
import logging

class KnowledgeAgent(BaseAgent):
    """
    Knowledge Agent:
    Uses the existing AffectraRetriever (RAG) to find factual context.
    It does NOT implement a second retriever.
    """
    def __init__(self, retriever: AffectraRetriever = None):
        # Allow passing an existing retriever instance (useful for dependency injection/lazy loading)
        self.retriever = retriever if retriever else AffectraRetriever()

    @property
    def name(self) -> str:
        return "KnowledgeAgent"

    def execute(self, context: AgentContext) -> AgentContext:
        # If there's no specific user query, maybe use the emotion label to query?
        # For a standard RAG query, user_query is required. 
        # For explanations, we might use the emotion label if user_query is missing.
        query = context.user_query
        if not query:
            if context.emotion_label and context.sentiment_label:
                query = f"What is {context.emotion_label} emotion and {context.sentiment_label} sentiment?"
            else:
                context.retrieved_context_str = "No query provided for knowledge retrieval."
                context.retrieved_context_items = []
                return context

        try:
            results = self.retriever.retrieve(query, top_k=context.top_k)
            context_items = []
            rag_context_str = ""
            
            if not results:
                rag_context_str = "No relevant context found in the knowledge base."
            else:
                for res in results:
                    context_items.append({
                        "source": res.source,
                        "chunk_index": res.chunk_index,
                        "score": res.score,
                        "text": res.text
                    })
                    rag_context_str += f"[Source: {res.source}, Chunk: {res.chunk_index}]\n{res.text}\n\n"
                    
            context.retrieved_context_str = rag_context_str.strip()
            context.retrieved_context_items = context_items
            
        except Exception as e:
            logging.error(f"KnowledgeAgent Retrieval failed: {e}")
            context.retrieved_context_str = "Error retrieving context from knowledge base."
            context.retrieved_context_items = []

        return context
