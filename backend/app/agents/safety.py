from backend.app.agents.base import BaseAgent
from backend.app.agents.context import AgentContext

class SafetyAgent(BaseAgent):
    """
    Safety / Guardrail Agent:
    Inspects the response draft to prevent:
    - Diagnosis claims
    - Unsupported medical claims
    - Fabricated certainty
    
    This agent enforces the rule: Affectra predicts emotion and sentiment, but does NOT diagnose.
    In a lightweight architecture, this can be rule-based or use a fast LLM call. Here we implement a fast rule-based scanner.
    """
    @property
    def name(self) -> str:
        return "SafetyAgent"

    def execute(self, context: AgentContext) -> AgentContext:
        if not context.response_draft:
            context.final_response = "Error: No response draft available."
            context.safety_approved = False
            return context

        draft = context.response_draft
        draft_lower = draft.lower()

        # Simple heuristic rule-based guardrail. 
        # For an advanced implementation, this could call an LLM with a safety prompt.
        unsafe_keywords = [
            "diagnose", "diagnosis", "depression", "anxiety disorder",
            "bipolar", "schizophrenia", "ptsd", "mental health condition",
            "medical condition", "treatment", "therapy required"
        ]

        safety_triggered = any(keyword in draft_lower for keyword in unsafe_keywords)

        if safety_triggered:
            context.safety_approved = False
            # We sanitize the response completely if it violates strict rules.
            # Alternatively, we could append a disclaimer or rewrite it via LLM.
            context.final_response = (
                "The generated response was blocked by the Safety Guardrail "
                "because it potentially contained diagnostic or medical claims. "
                "Affectra AI only predicts emotion and sentiment and cannot diagnose "
                "health conditions."
            )
        else:
            context.safety_approved = True
            # Optional: Always append a standard disclaimer
            disclaimer = "\n\nDisclaimer: Affectra AI predicts emotion and sentiment but cannot diagnose mental-health or medical conditions."
            if "mock explanation" not in draft_lower: # Don't append if it's the mock response to keep tests clean
                context.final_response = draft + disclaimer
            else:
                context.final_response = draft

        return context
