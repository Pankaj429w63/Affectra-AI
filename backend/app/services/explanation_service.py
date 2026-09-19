from backend.app.schemas.explanation import ExplanationRequest, ExplanationResponse
from backend.app.llm.explanation import get_llm_provider
from backend.app.llm.prompts import EXPLANATION_PROMPT_TEMPLATE
from backend.app.core.config import settings

class ExplanationService:
    def __init__(self):
        self.provider = get_llm_provider()
        
        # Initialize RAG Pipeline lazily to avoid startup overhead if disabled
        self.rag_pipeline = None
        if settings.RAG_ENABLED:
            from rag.pipeline import AffectraRAG
            self.rag_pipeline = AffectraRAG()

    def generate_explanation(self, request: ExplanationRequest) -> ExplanationResponse:
        emo_probs_str = "\n".join([f"- {k}: {v:.4f}" for k, v in request.emotion_probabilities.items()])
        sent_probs_str = "\n".join([f"- {k}: {v:.4f}" for k, v in request.sentiment_probabilities.items()])

        # Retrieve Context via RAG if enabled
        rag_context_str = "No retrieved context available."
        if settings.RAG_ENABLED and self.rag_pipeline:
            # We query the RAG pipeline using the predicted emotion label to get definition/context
            query = f"What is {request.emotion_label} emotion and {request.sentiment_label} sentiment?"
            try:
                rag_result = self.rag_pipeline.query(query)
                if rag_result["formatted_context"]:
                    rag_context_str = rag_result["formatted_context"]
            except Exception as e:
                print(f"Warning: RAG retrieval failed: {e}")

        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            emotion_label=request.emotion_label,
            emotion_probs=emo_probs_str,
            sentiment_label=request.sentiment_label,
            sentiment_probs=sent_probs_str,
            rag_context=rag_context_str
        )

        try:
            explanation_text = self.provider.generate(prompt)
        except Exception as e:
            explanation_text = f"An error occurred while generating the explanation: {str(e)}"

        return ExplanationResponse(explanation=explanation_text)

explanation_service = ExplanationService()
