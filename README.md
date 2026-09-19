<div align="center">
  <img src="assets/dashboard.png" alt="Affectra AI Dashboard" width="100%">
  
  <br>
  
 Affectra AI

Production-oriented multimodal emotion and sentiment intelligence platform combining text, audio, and video with gated fusion, explainable AI, RAG, and agent orchestration.

Live Demo: https://affectra-ai.vercel.app
API: https://affectra-ai.onrender.com
API Docs: https://affectra-ai.onrender.com/docs
GitHub: https://github.com/Pankaj429w63/Affectra-AI

Overview

Affectra AI is an end-to-end multimodal AI system that combines linguistic, acoustic, and visual information to predict emotion and sentiment.

The system uses pretrained encoders for each modality, maps them into a common 768-dimensional representation, and combines them using a custom Gated Multimodal Fusion network.

Outputs

7 emotions: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

3 sentiments: Positive, Negative, Neutral

The classifier is intentionally separated from the generative AI layer: the ML model produces predictions, while LLM, RAG, and agents provide explanations, grounded knowledge, and interaction.

Architecture

Text ──► DistilRoBERTa ──► 768 ──┐
Audio ─► Wav2Vec2      ──► 768 ──┼─► Gated Multimodal Fusion ─► Emotion
Video ─► ViT           ──► 768 ──┘          512                └► Sentiment
                                                    │
                                                    ▼
                                               FastAPI
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                         LLM Explanation          RAG                 Agents
                              │                     │                     │
                              └─────────────────────┼─────────────────────┘
                                                    ▼
                                             React + Vite UI

Technology Stack

Layer

Technology

Language

Python, TypeScript

ML

PyTorch, Hugging Face Transformers

Text

distilroberta-base

Audio

facebook/wav2vec2-base

Video

google/vit-base-patch16-224

Fusion

Custom Gated Multimodal Fusion

Backend

FastAPI, Pydantic

Frontend

React 18, Vite, Tailwind CSS

RAG

FAISS, SentenceTransformers

LLM

Provider-agnostic explanation layer

Agents

Interpretation, Knowledge, Response, Safety

Deployment

Docker, Render, Vercel

Artifact Storage

Private Hugging Face repository

Dataset

Training and evaluation use the MELD (Multimodal EmotionLines Dataset).

Split

Samples

Train

9,989

Development

1,109

Test

2,610

Total

13,708

Raw datasets, feature caches, large checkpoints, and private artifacts are intentionally excluded from the public repository.

Final Model Results

The selected production experiment is Experiment 2.

Emotion

Metric

Test

Accuracy

55.29%

Weighted Precision

57.36%

Weighted Recall

55.29%

Weighted F1

56.03%

Macro F1

37.01%

Sentiment

Metric

Test

Accuracy

67.16%

Weighted F1

67.01%

Macro F1

64.16%

Emotion F1

Emotion

F1

Anger

36.70%

Disgust

8.64%

Fear

9.72%

Joy

51.21%

Neutral

72.97%

Sadness

29.77%

Surprise

50.08%

The class-level results are reported transparently because emotion datasets are imbalanced and minority classes remain challenging.

Key Engineering Features

Multimodal text + audio + video representation learning

Frozen pretrained encoders with compact trainable fusion

Custom gated multimodal fusion network (~594K trainable parameters)

Separate emotion and sentiment prediction heads

FastAPI inference service

Strict Pydantic request validation

Provider-agnostic LLM explanation layer

FAISS-based RAG knowledge retrieval

Specialized agent orchestration

Safety guardrails for generated responses

React/TypeScript production dashboard

Dockerized backend

Vercel + Render deployment

Lazy loading for memory-constrained deployment

Production environment configuration and CORS handling

API

Endpoint

Method

Purpose

/health

GET

Service and model health

/predict

POST

Prediction from modality features

/predict/raw

POST

Raw media feature extraction + prediction

/explain

POST

Generate prediction explanation

/rag

POST

Knowledge-grounded retrieval

/agents

POST

Agent orchestration

/docs

GET

Swagger documentation

Interactive API documentation:

https://affectra-ai.onrender.com/docs

RAG Pipeline

Knowledge Documents
       ↓
Chunking
       ↓
SentenceTransformer
       ↓
384-D Embeddings
       ↓
FAISS IndexFlatIP
       ↓
Top-K Retrieval
       ↓
LLM / Agent Context
       ↓
Grounded Response

Agent Pipeline

Interpretation Agent
        ↓
Knowledge Agent
        ↓
Response Agent
        ↓
Safety Agent

The agent layer does not replace the trained classifier. It operates after prediction to interpret results, retrieve relevant knowledge, formulate responses, and apply safety constraints.

Repository Structure

Affectra-AI/
├── assets/                 # UI and README assets
├── backend/                # FastAPI application
├── frontend/               # React + Vite application
├── training/               # Training and inference pipeline
├── models/                 # Model configuration/artifact references
├── rag/                    # RAG ingestion and retrieval
├── docs/                   # Architecture and training documentation
├── scripts/                # Experiment/deployment/validation scripts
├── tests/                  # Tests
├── render.yaml             # Render configuration
├── .dockerignore
├── .env.example
├── .gitignore
└── README.md

Local Setup

Prerequisites

Python 3.11+

Node.js 18+

npm

Git

Clone

git clone https://github.com/Pankaj429w63/Affectra-AI.git
cd Affectra-AI

Backend

python -m venv .venv

Windows:

.venv\Scripts\activate

Linux/macOS:

source .venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

Swagger:

http://127.0.0.1:8000/docs

Frontend

cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173

Configure the API through:

VITE_API_BASE_URL

Never commit secrets.

Deployment

                    Internet
                       │
                       ▼
             ┌─────────────────────┐
             │ Vercel              │
             │ React + Vite        │
             └──────────┬──────────┘
                        │ HTTPS
                        ▼
             ┌─────────────────────┐
             │ Render              │
             │ FastAPI + Docker    │
             └──────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        ML/RAG artifacts       API services
        Hugging Face           Prediction/RAG/
                               Explanation/Agents

Production Limitation

The current Render free instance has a 512 MB memory limit. The /predict/raw endpoint can exceed this limit because raw multimodal processing loads heavy pretrained text, audio, and video encoders.

This is a deployment-resource limitation, not a training failure.

The trained fusion model remains unchanged. Reliable raw multimodal production inference should use a higher-memory inference instance or a more aggressively optimized/quantized extraction service.

Validation

The project has been validated across:

ML evaluation and production artifact verification

FastAPI health and prediction APIs

LLM explanation flow

RAG retrieval

agent orchestration

React production build

SPA routing

frontend/backend integration

CORS configuration

Docker/Render deployment configuration

Vercel production deployment

Responsible AI

Affectra AI is an engineering/research system and is not a clinical or psychological diagnostic tool.

Predictions represent model outputs and should not be treated as definitive statements about a person's internal mental state.

The system keeps prediction and generation separate so that an LLM cannot silently change the classifier's predicted labels.

Security

The repository intentionally excludes:

API keys and access tokens

private deployment credentials

.env files containing secrets

raw datasets/media

large feature caches

unnecessary checkpoints

Secrets are supplied through deployment environment variables.

Documentation

See:

docs/SYSTEM_DESIGN.md

docs/TRAINING_ARCHITECTURE.md

docs/MIGRATION_PLAN.md

Future Work

Higher-memory raw multimodal inference

Encoder quantization

Asynchronous media processing

Better minority-class performance

Calibration and uncertainty estimation

Multimodal attention visualization

Experiment tracking

CI/CD automation

Production monitoring and observability

Scalable inference workers

Author

Pankaj Yadav

AI/ML | Multimodal AI | Deep Learning | Generative AI

GitHub: https://github.com/Pankaj429w63
LinkedIn: https://www.linkedin.com/in/pankaj-yadav-0172162a2/

License

MIT License. See LICENSE.

More Understanding. A Kinder Tomorrow.
