# Affectra AI API

This is the FastAPI backend for the Affectra AI Multimodal Emotion Intelligence Platform. 
It loads the production `AffectraPredictor` (Experiment 2) and exposes a simple REST API for inference.

## Setup

1. **Navigate to the repository root** and create a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install the backend requirements**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Start the FastAPI server**:
   Make sure you run this command from the *root of the repository* (not inside the `backend` folder) so that the relative imports to `training` work properly.
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Endpoints

### 1. Health Check
`GET /health`
Returns the status of the service and whether the model is successfully loaded.
```json
{
  "status": "healthy",
  "service": "Affectra AI API",
  "model_loaded": true
}
```

### 2. Predict
`POST /predict`
Performs multimodal inference on given feature vectors.

**Request Body (JSON)**:
```json
{
  "text_feat": [0.0, 0.1, ...], // 768 elements
  "audio_feat": [0.0, 0.1, ...], // 768 elements
  "video_feat": [0.0, 0.1, ...]  // 768 elements
}
```

**Response**:
```json
{
  "emotion": {
    "label": "neutral",
    "probabilities": {
      "anger": 0.05,
      "disgust": 0.02,
      "fear": 0.01,
      "joy": 0.05,
      "neutral": 0.8,
      "sadness": 0.04,
      "surprise": 0.03
    }
  },
  "sentiment": {
    "label": "neutral",
    "probabilities": {
      "positive": 0.1,
      "negative": 0.1,
      "neutral": 0.8
    }
  }
}

### 3. Explain
`POST /explain`
Generates a grounded explanation of the ML model's prediction.

### 4. RAG
`POST /rag`
Retrieves relevant knowledge base context and generates a grounded answer to a user question using the LLM.

**Request Body (JSON)**:
```json
{
  "question": "What emotions does Affectra AI recognize?",
  "top_k": 3
}
```

**Response**:
```json
{
  "question": "What emotions does Affectra AI recognize?",
  "answer": "Affectra AI recognizes seven basic emotions: anger, disgust, fear, joy, neutral, sadness, and surprise.",
  "retrieved_context": [
    {
      "source": "emotion_knowledge.md",
      "chunk_index": 1,
      "score": 0.82,
      "text": "..."
    }
  ]
}
```

## LLM Mock Mode Setup
For local testing without an API key, set `LLM_PROVIDER=mock` in your `.env` file (or environment). The mock provider will return a deterministic response without making external network calls.

## Running the Server
Use PowerShell to start the uvicorn daemon:
```powershell
uvicorn backend.app.main:app --port 8000
```

## Final RAG Status (Phase 4.7)

The RAG architecture is completely validated and implemented:
- **Knowledge Base**: Implemented
- **Chunking**: Implemented
- **Embeddings**: Implemented
- **FAISS**: Implemented
- **Retriever**: Implemented
- **RAG Service**: Implemented
- **FastAPI /rag**: Implemented
- **LLM-RAG Integration**: Implemented
- **Complete Validation**: Implemented

**Current Validation uses MockLLMProvider.** The real LLM grounding behavior has not been fully validated (it relies on the external LLM to obey the prompt) because testing the Mock provider guarantees isolated local testing without API keys, but the Mock provider relies on deterministic static responses.

**Features Not Yet Implemented:**
- Agents
- Frontend
- Deployment

### Useful Commands
```powershell
# Run backend
uvicorn backend.app.main:app --port 8000

# Open Swagger
Start-Process "http://localhost:8000/docs"

# Run complete RAG validation
$env:PYTHONPATH="."; python validate_complete_rag.py

# Run RAG unit tests
$env:PYTHONPATH="."; python -m pytest tests/test_rag.py tests/test_rag_api.py tests/test_retriever.py -v
```

---

## Final Agent API Status (Phase 5.3)

The internal agent orchestrator is now securely exposed via a dedicated FastAPI endpoint.

### `POST /agents`
Executes the unified Affectra AI Agent pipeline to process predictions and generate grounded explanations.

#### Execution Flow
1. **InterpretationAgent**: Formats raw ML labels and probabilities.
2. **KnowledgeAgent**: Retrieves factual chunks via the `AffectraRetriever` (RAG).
3. **ResponseAgent**: Drafts a conversational response grounded in the retrieved facts using the `LLMProvider`.
4. **SafetyAgent**: Sanitizes responses containing prohibited medical/diagnostic language.

#### Example Request
```json
{
  "user_query": "What emotions does Affectra AI recognize?",
  "emotion_label": "joy",
  "emotion_probabilities": {"joy": 0.8, "sadness": 0.2},
  "sentiment_label": "positive",
  "sentiment_probabilities": {"positive": 0.9, "negative": 0.1},
  "top_k": 3
}
```

#### Example Response
```json
{
  "final_response": "Affectra AI recognizes emotions such as joy, sadness, anger, fear, surprise, and disgust.\n\nDisclaimer: Affectra AI predicts emotion and sentiment but cannot diagnose mental-health or medical conditions.",
  "safety_approved": true,
  "execution_trace": [
    "InterpretationAgent (success, 0.001s)",
    "KnowledgeAgent (success, 0.024s)",
    "ResponseAgent (success, 1.205s)",
    "SafetyAgent (success, 0.001s)"
  ],
  "retrieved_context": [
    {
      "source": "emotion_knowledge.md",
      "chunk_index": 0,
      "score": 0.85,
      "text": "Affectra AI recognizes..."
    }
  ],
  "error_message": null
}
```

### Useful Agent Commands
```powershell
# Run backend
uvicorn backend.app.main:app --port 8000

# Open Swagger for Agent Endpoint Documentation
Start-Process "http://localhost:8000/docs"

# Run agent pipeline API validation (Real prediction-style logic)
$env:PYTHONPATH="."; python validate_agents_api.py

# Run agent API unit tests
$env:PYTHONPATH="."; python -m pytest tests/test_agents_api.py -v
```
