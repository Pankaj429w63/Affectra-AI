import os
import pytest
from backend.app.agents.context import AgentContext
from backend.app.agents.interpretation import InterpretationAgent
from backend.app.agents.knowledge import KnowledgeAgent
from backend.app.agents.response import ResponseAgent
from backend.app.agents.safety import SafetyAgent
from backend.app.llm.provider import MockLLMProvider
from rag.retriever import AffectraRetriever

@pytest.fixture(scope="module", autouse=True)
def setup_mock_provider():
    os.environ["LLM_PROVIDER"] = "mock"
    yield

def test_agent_context_initialization():
    ctx = AgentContext()
    assert ctx.top_k == 3
    assert ctx.safety_approved is True
    assert ctx.user_query is None

def test_interpretation_agent():
    agent = InterpretationAgent()
    assert agent.name == "InterpretationAgent"
    
    ctx = AgentContext(
        emotion_label="joy",
        emotion_probabilities={"joy": 0.9, "sadness": 0.1},
        sentiment_label="positive",
        sentiment_probabilities={"positive": 0.8, "negative": 0.2}
    )
    
    ctx = agent.execute(ctx)
    assert ctx.interpretation_result is not None
    assert "joy" in ctx.interpretation_result
    assert "0.9" in ctx.interpretation_result
    assert "positive" in ctx.interpretation_result

def test_interpretation_agent_empty_context():
    agent = InterpretationAgent()
    ctx = AgentContext()
    ctx = agent.execute(ctx)
    assert "No valid emotion or sentiment prediction" in ctx.interpretation_result

def test_knowledge_agent_no_query():
    # If no query and no prediction, it should return early safely
    agent = KnowledgeAgent()
    ctx = AgentContext()
    ctx = agent.execute(ctx)
    assert "No query provided" in ctx.retrieved_context_str

def test_knowledge_agent_with_query():
    # This assumes the FAISS index exists locally.
    agent = KnowledgeAgent()
    agent.retriever.load() # ensure loaded
    
    ctx = AgentContext(user_query="What emotions does Affectra AI recognize?")
    ctx = agent.execute(ctx)
    
    assert ctx.retrieved_context_str is not None
    assert len(ctx.retrieved_context_items) > 0

def test_response_agent():
    mock_provider = MockLLMProvider()
    agent = ResponseAgent(provider=mock_provider)
    
    ctx = AgentContext(
        user_query="Hello",
        interpretation_result="Interpreted text",
        retrieved_context_str="Retrieved context"
    )
    
    ctx = agent.execute(ctx)
    assert ctx.response_draft is not None
    # We expect the mock provider to return its deterministic text
    assert "mock explanation" in ctx.response_draft.lower()

def test_safety_agent_pass():
    agent = SafetyAgent()
    ctx = AgentContext(response_draft="The user looks very happy based on the facial features.")
    ctx = agent.execute(ctx)
    
    assert ctx.safety_approved is True
    # The safety agent appends a disclaimer unless it's a "mock explanation" exactly
    assert "Disclaimer" in ctx.final_response

def test_safety_agent_mock_pass():
    agent = SafetyAgent()
    # "mock explanation" strings skip the disclaimer to keep tests clean
    ctx = AgentContext(response_draft="This is a mock explanation.")
    ctx = agent.execute(ctx)
    
    assert ctx.safety_approved is True
    assert "mock explanation" in ctx.final_response.lower()
    assert "Disclaimer" not in ctx.final_response

def test_safety_agent_blocked():
    agent = SafetyAgent()
    # Include an unsafe keyword: "diagnosis"
    ctx = AgentContext(response_draft="My diagnosis is that the user has depression.")
    ctx = agent.execute(ctx)
    
    assert ctx.safety_approved is False
    assert "Safety Guardrail" in ctx.final_response
    assert "diagnosis" not in ctx.final_response # Sanitized
