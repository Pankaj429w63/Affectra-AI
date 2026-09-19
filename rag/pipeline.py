from rag.retriever import AffectraRetriever

class AffectraRAG:
    def __init__(self):
        self.retriever = AffectraRetriever()

    def query(self, query: str, top_k: int = None):
        """
        Executes a retrieval query and constructs a grounded context block.
        Does NOT call an LLM directly.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        
        context = ""
        for i, res in enumerate(results):
            context += f"--- Source {i+1} ({res['source']}) ---\n"
            context += f"{res['text']}\n\n"
            
        return {
            "query": query,
            "results": results,
            "formatted_context": context.strip()
        }
