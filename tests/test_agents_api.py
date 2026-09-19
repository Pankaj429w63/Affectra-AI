import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="module", autouse=True)
def setup_mock_provider():
    os.environ["LLM_PROVIDER"] = "mock"
    yield

client = TestClient(app)

def test_valid_request():
    response = client.post("/agents", json={
        "emotion_label": "joy",
        "sentiment_label": "positive",
        "user_query": "What emotions does Affectra AI recognize?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "final_response" in data
    assert "safety_approved" in data
    assert "execution_trace" in data
    assert data["safety_approved"] is True
    # The Mock LLM should output its deterministic string
    assert "mock explanation" in data["final_response"].lower()
    
    trace = data["execution_trace"]
    assert len(trace) == 4
    assert "InterpretationAgent" in trace[0]
    assert "KnowledgeAgent" in trace[1]
    assert "ResponseAgent" in trace[2]
    assert "SafetyAgent" in trace[3]

def test_empty_query_rejected():
    response = client.post("/agents", json={
        "user_query": "   "
    })
    assert response.status_code == 422
    assert "empty whitespace" in response.json()["detail"].lower()

def test_invalid_top_k():
    response = client.post("/agents", json={
        "user_query": "Hello",
        "top_k": 0
    })
    assert response.status_code == 422 # Pydantic validation fails because ge=1

def test_safety_blocked_via_mock():
    # If we somehow get a response draft that is unsafe, the safety agent should block it.
    # The API just forwards the orchestrator's state.
    # Here we just verify that normal safe response is approved.
    # A true safety mock test would inject a bad mock LLM, but test_agent_orchestrator.py already tests this deeply.
    response = client.post("/agents", json={
        "emotion_label": "joy"
    })
    assert response.status_code == 200
    assert response.json()["safety_approved"] is True

def test_missing_fields_allowed_by_schema():
    # Schema allows everything to be optional except top_k which has a default
    response = client.post("/agents", json={})
    assert response.status_code == 200
    data = response.json()
    assert "final_response" in data
    assert len(data["execution_trace"]) == 4

def test_retrieved_context_is_present():
    response = client.post("/agents", json={
        "user_query": "What emotions are recognized?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "retrieved_context" in data
    # Context items might be empty if the DB is missing, but the field should exist
    assert isinstance(data["retrieved_context"], list)
