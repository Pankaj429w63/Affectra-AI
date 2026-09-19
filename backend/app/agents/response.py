from backend.app.agents.base import BaseAgent
from backend.app.agents.context import AgentContext
from backend.app.llm.explanation import get_llm_provider
from backend.app.llm.provider import LLMProvider
import logging

RESPONSE_PROMPT_TEMPLATE = """
You are the Response Agent for Affectra AI.

--- ML Prediction Interpretation ---
{interpretation}

--- Retrieved Factual Knowledge ---
{rag_context}

--- User Query ---
{user_query}

Your task is to combine the ML interpretation, the factual knowledge, and the user query to draft a human-friendly response.
If the user did not provide a query, just provide a clear explanation of the ML prediction grounded in the factual knowledge.

CRITICAL RULES:
1. Ground your answer in the Retrieved Factual Knowledge.
2. Do not invent facts.
3. Accept the ML Prediction Interpretation as truth. Do not alter the emotion or sentiment labels.
4. Do not diagnose mental health or medical conditions.

Draft the response:
"""

class ResponseAgent(BaseAgent):
    """
    Response Agent:
    Combines interpretation, retrieved context, and user input to draft a final response.
    Uses the provider-agnostic LLMProvider.
    """
    def __init__(self, provider: LLMProvider = None):
        self.provider = provider if provider else get_llm_provider()

    @property
    def name(self) -> str:
        return "ResponseAgent"

    def execute(self, context: AgentContext) -> AgentContext:
        interpretation = context.interpretation_result or "No ML prediction provided."
        rag_context = context.retrieved_context_str or "No factual context retrieved."
        user_query = context.user_query or "No specific user query."

        prompt = RESPONSE_PROMPT_TEMPLATE.format(
            interpretation=interpretation,
            rag_context=rag_context,
            user_query=user_query
        )

        try:
            draft = self.provider.generate(prompt)
            context.response_draft = draft
        except Exception as e:
            logging.error(f"ResponseAgent failed to generate draft: {e}")
            context.response_draft = "An error occurred while drafting the response."

        return context
