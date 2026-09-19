import os
import json
from backend.app.agents.context import AgentContext
from backend.app.agents.orchestrator import AffectraAgentOrchestrator

def print_result(scenario: str, context: AgentContext):
    print("=" * 60)
    print(f"SCENARIO: {scenario}")
    print("=" * 60)
    print(f"Input Emotion: {context.emotion_label}")
    print(f"Input Sentiment: {context.sentiment_label}")
    print(f"User Query: {context.user_query}")
    print("-" * 60)
    print("EXECUTION TRACE:")
    for step in context.execution_trace:
        print(f"  -> {step}")
    print("-" * 60)
    print(f"SAFETY APPROVED: {context.safety_approved}")
    if context.error_message:
        print(f"ERROR: {context.error_message}")
    print("\nFINAL RESPONSE:")
    print(context.final_response)
    print("\n")

def run_validations():
    # Force Mock Provider for deterministic validation without API Keys
    os.environ["LLM_PROVIDER"] = "mock"
    
    print("Initializing Affectra Agent Orchestrator...")
    orchestrator = AffectraAgentOrchestrator()
    print("Orchestrator initialized successfully.\n")

    # Scenario 1: Normal emotion prediction, no query
    ctx1 = AgentContext(
        emotion_label="joy",
        emotion_probabilities={"joy": 0.85, "neutral": 0.1, "sadness": 0.05},
        sentiment_label="positive",
        sentiment_probabilities={"positive": 0.9, "neutral": 0.08, "negative": 0.02}
    )
    result1 = orchestrator.execute(ctx1)
    print_result("1. Normal Emotion Prediction (No Query)", result1)

    # Scenario 2: Different emotion prediction, no query
    ctx2 = AgentContext(
        emotion_label="anger",
        emotion_probabilities={"anger": 0.75, "disgust": 0.2, "neutral": 0.05},
        sentiment_label="negative",
        sentiment_probabilities={"negative": 0.95, "neutral": 0.05}
    )
    result2 = orchestrator.execute(ctx2)
    print_result("2. Different Emotion Prediction (No Query)", result2)

    # Scenario 3: Mixed/Neutral sentiment prediction
    ctx3 = AgentContext(
        emotion_label="neutral",
        emotion_probabilities={"neutral": 0.9, "sadness": 0.05, "joy": 0.05},
        sentiment_label="neutral",
        sentiment_probabilities={"neutral": 0.9, "positive": 0.05, "negative": 0.05}
    )
    result3 = orchestrator.execute(ctx3)
    print_result("3. Neutral Prediction (No Query)", result3)

    # Scenario 4: User question requiring RAG knowledge
    ctx4 = AgentContext(
        emotion_label="surprise",
        emotion_probabilities={"surprise": 0.8},
        sentiment_label="positive",
        sentiment_probabilities={"positive": 0.7},
        user_query="What emotions does Affectra AI recognize?"
    )
    result4 = orchestrator.execute(ctx4)
    print_result("4. User Query + RAG Integration", result4)

if __name__ == "__main__":
    run_validations()
