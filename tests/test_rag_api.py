import os
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.rag import RAGRequest
from backend.app.services.rag_service import rag_service

@pytest.fixture(scope="module", autouse=True)
def setup_mock_provider():
    os.environ["LLM_PROVIDER"] = "mock"
    # Ensure retriever is loaded to avoid delays during tests
    rag_service.load()
    yield

@pytest.fixture
def client():
    return TestClient(app)

def test_valid_rag_question(client):
    response = client.post("/rag", json={"question": "What is Affectra AI?", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert "retrieved_context" in data
    assert len(data["retrieved_context"]) == 2
    assert "mock explanation" in data["answer"].lower() or "mock" in data["answer"].lower()

def test_empty_question_rejected(client):
    response = client.post("/rag", json={"question": "", "top_k": 2})
    assert response.status_code == 422

def test_whitespace_question_rejected(client):
    response = client.post("/rag", json={"question": "   \t\n  ", "top_k": 2})
    assert response.status_code == 422

def test_invalid_top_k_rejected(client):
    response = client.post("/rag", json={"question": "Test", "top_k": 0})
    assert response.status_code == 422
    
    response = client.post("/rag", json={"question": "Test", "top_k": -5})
    assert response.status_code == 422

def test_response_preserves_metadata(client):
    response = client.post("/rag", json={"question": "Metadata test", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    ctx = data["retrieved_context"][0]
    assert "source" in ctx
    assert "chunk_index" in ctx
    assert "score" in ctx
    assert "text" in ctx

def test_unsupported_question_behavior(client):
    # This just tests that it successfully runs through the mock provider. 
    # Real "I don't know" logic requires a real LLM.
    response = client.post("/rag", json={"question": "What is the capital of France?", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
