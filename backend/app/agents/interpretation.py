from backend.app.agents.base import BaseAgent
from backend.app.agents.context import AgentContext

class InterpretationAgent(BaseAgent):
    """
    Interpretation Agent:
    Transforms raw ML predictions into structured, human-readable text for downstream agents.
    It does NOT perform emotion classification itself.
    """
    @property
    def name(self) -> str:
        return "InterpretationAgent"

    def execute(self, context: AgentContext) -> AgentContext:
        if not context.emotion_label or not context.sentiment_label:
            context.interpretation_result = "No valid emotion or sentiment prediction provided by the ML model."
            return context

        emo_probs_str = ""
        if context.emotion_probabilities:
            emo_probs_str = "\n".join([f"- {k}: {v:.4f}" for k, v in context.emotion_probabilities.items()])
            
        sent_probs_str = ""
        if context.sentiment_probabilities:
            sent_probs_str = "\n".join([f"- {k}: {v:.4f}" for k, v in context.sentiment_probabilities.items()])

        interpretation = (
            f"Predicted Emotion: {context.emotion_label}\n"
            f"Emotion Confidence Scores:\n{emo_probs_str}\n\n"
            f"Predicted Sentiment: {context.sentiment_label}\n"
            f"Sentiment Confidence Scores:\n{sent_probs_str}"
        )
        
        context.interpretation_result = interpretation
        return context
