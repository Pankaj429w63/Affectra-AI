import os
import pytest
from backend.app.agents.context import AgentContext
from backend.app.agents.orchestrator import AffectraAgentOrchestrator
from backend.app.llm.provider import MockLLMProvider
from backend.app.agents.base import BaseAgent

@pytest.fixture(scope="module", autouse=True)
def setup_mock_provider():
    os.environ["LLM_PROVIDER"] = "mock"
    yield

def test_successful_complete_pipeline():
    mock_provider = MockLLMProvider()
    orchestrator = AffectraAgentOrchestrator(provider=mock_provider)
    
    context = AgentContext(
        user_query="What emotions does Affectra AI recognize?",
        emotion_label="joy",
        emotion_probabilities={"joy": 0.9, "sadness": 0.1},
        sentiment_label="positive",
        sentiment_probabilities={"positive": 0.8, "negative": 0.2}
    )
    
    result = orchestrator.execute(context)
    
    # Verify outputs
    assert result.interpretation_result is not None
    assert "joy" in result.interpretation_result
    
    assert result.retrieved_context_str is not None
    assert "mock explanation" in result.response_draft.lower()
    
    assert result.safety_approved is True
    assert result.final_response is not None
    assert "mock explanation" in result.final_response.lower()

    # Trace Verification
    assert len(result.execution_trace) == 4
    assert "InterpretationAgent" in result.execution_trace[0]
    assert "KnowledgeAgent" in result.execution_trace[1]
    assert "ResponseAgent" in result.execution_trace[2]
    assert "SafetyAgent" in result.execution_trace[3]

def test_safety_blocked_pipeline():
    # Force the Mock provider to return a string that triggers safety
    class BadMockProvider(MockLLMProvider):
        def generate(self, prompt: str, **kwargs) -> str:
            return "My diagnosis is that you have depression."

    orchestrator = AffectraAgentOrchestrator(provider=BadMockProvider())
    context = AgentContext(
        emotion_label="sadness",
        emotion_probabilities={"sadness": 0.9},
        sentiment_label="negative",
        sentiment_probabilities={"negative": 0.9}
    )
    
    result = orchestrator.execute(context)
    
    assert result.safety_approved is False
    assert "Safety Guardrail" in result.final_response
    assert "diagnosis" not in result.final_response  # Replaced with safe text

class FailingAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "FailingAgent"
    def execute(self, context: AgentContext) -> AgentContext:
        raise ValueError("Simulated random internal failure")

def test_error_handling_pipeline():
    orchestrator = AffectraAgentOrchestrator(provider=MockLLMProvider())
    
    # Inject a failing agent in the middle of the pipeline
    orchestrator.pipeline.insert(2, FailingAgent())
    
    context = AgentContext(emotion_label="anger")
    result = orchestrator.execute(context)
    
    # Should safely fail without throwing exception out of orchestrator
    assert result.safety_approved is False
    assert result.error_message == "An internal error occurred during FailingAgent execution."
    assert "unexpected error" in result.final_response
    
    # Trace should indicate failure
    assert "FailingAgent (FAILED)" in result.execution_trace[-1]

def test_no_query_pipeline():
    # Pipeline without query should still formulate an explanation
    orchestrator = AffectraAgentOrchestrator(provider=MockLLMProvider())
    context = AgentContext(
        emotion_label="neutral",
        sentiment_label="neutral"
    )
    
    result = orchestrator.execute(context)
    assert result.safety_approved is True
    assert result.final_response is not None
    assert len(result.execution_trace) == 4
