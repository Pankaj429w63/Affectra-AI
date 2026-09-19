<div align="center">
  <img src="assets/dashboard.png" alt="Affectra AI Dashboard" width="100%">
  
  <br>
  
from pathlib import Path

readme = r"""# Affectra AI

> **Production-oriented multimodal emotion and sentiment intelligence platform combining text, audio, and video with gated fusion, explainable AI, RAG, and agent orchestration.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)](https://vite.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live Demo:** https://affectra-ai.vercel.app  
**API:** https://affectra-ai.onrender.com  
**API Docs:** https://affectra-ai.onrender.com/docs  
**Source:** https://github.com/Pankaj429w63/Affectra-AI

---

## Overview

Affectra AI is an end-to-end **multimodal emotion and sentiment analysis system** designed to demonstrate how modern ML systems can move from model development to an integrated production architecture.

Instead of relying on a single modality, Affectra combines:

- **Text** — linguistic and semantic information
- **Audio** — speech/acoustic information
- **Video** — visual information

Each modality is converted into a fixed 768-dimensional representation using a pretrained encoder. These representations are then combined by a **Gated Multimodal Fusion** network to produce:

- **7 emotion classes:** Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
- **3 sentiment classes:** Positive, Negative, Neutral

The prediction layer is deliberately separated from the generative layer: the trained classifier produces the prediction, while LLM/RAG/agent components are used for explanation, knowledge grounding, and interaction.

This separation makes the system easier to test, reason about, and extend.

---

## Why This Project Matters

Affectra AI demonstrates a complete ML engineering workflow rather than only a model-training notebook:

**Dataset → Feature Extraction → Multimodal Fusion → Evaluation → Model Export → API → RAG → Agents → Frontend → Cloud Deployment**

The project covers practical engineering concerns including:

- pretrained Transformer integration
- multimodal representation learning
- gated feature fusion
- multi-task classification
- reproducible evaluation
- API design and validation
- retrieval-augmented generation
- agent orchestration
- frontend/backend integration
- Docker deployment
- cloud deployment
- memory-aware inference
- production configuration and CORS

---

## System Architecture

```text
                    ┌───────────────────────────┐
                    │       User Interface      │
                    │     React + TypeScript    │
                    │        + Vite             │
                    └─────────────┬─────────────┘
                                  │ HTTPS
                                  ▼
                    ┌───────────────────────────┐
                    │       FastAPI API         │
                    │                           │
                    │ /predict                 │
                    │ /predict/raw             │
                    │ /explain                 │
                    │ /rag                     │
                    │ /agents                  │
                    │ /health                  │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
        Text Encoder        Audio Encoder        Video Encoder
     DistilRoBERTa-base     Wav2Vec2-base          ViT-base
          [768]                [768]                [768]
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ Gated Multimodal Fusion   │
                    │        ~594K params       │
                    │        Fusion: 512        │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │   Prediction Head         │
                    │                           │
                    │ Emotion: 7 classes        │
                    │ Sentiment: 3 classes      │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
       Explanation LLM        RAG Layer          Agent System
       interpretation       FAISS + KB          specialized agents
Machine Learning Pipeline
1. Pretrained Encoders
Modality	Encoder	Output
Text	distilroberta-base	768
Audio	facebook/wav2vec2-base	768
Video	google/vit-base-patch16-224	768

The encoders are used as frozen feature extractors during fusion training.

2. Feature Fusion

The three modality representations are passed to a custom Gated Multimodal Fusion network.

Text   ──► 768 ──┐
Audio  ──► 768 ──┼──► Gated Fusion ──► 512
Video  ──► 768 ──┘
                         │
                         ├──► Emotion classifier (7)
                         └──► Sentiment classifier (3)

The fusion model contains approximately 594K trainable parameters, making the trainable component substantially smaller than the pretrained encoders.

3. Prediction

The final production artifact is wrapped by AffectraPredictor.

The prediction system returns probability distributions rather than only a hard label, allowing the application layer to expose confidence information and supporting downstream explanation.

Dataset

The multimodal training pipeline is based on the MELD (Multimodal EmotionLines Dataset).

Dataset splits used by the project:

Split	Samples
Train	9,989
Development	1,109
Test	2,610
Total	13,708

MELD provides conversational multimodal data with emotion and sentiment annotations.

Dataset files, raw media, feature caches, and large checkpoints are intentionally excluded from the public Git repository.

Final Model Results

The selected production fusion experiment is Experiment 2.

Emotion
Metric	Test
Accuracy	55.29%
Weighted Precision	57.36%
Weighted Recall	55.29%
Weighted F1	56.03%
Macro F1	37.01%
Sentiment
Metric	Test
Accuracy	67.16%
Weighted F1	67.01%
Macro F1	64.16%
Emotion F1 by Class
Emotion	F1
Anger	36.70%
Disgust	8.64%
Fear	9.72%
Joy	51.21%
Neutral	72.97%
Sadness	29.77%
Surprise	50.08%

The macro/weighted metric difference reflects the class imbalance and the difficulty of learning minority emotion classes. This is an important limitation and is intentionally reported rather than hidden.

Explainability and Grounded AI

Affectra separates prediction from generation.

Prediction Layer

The multimodal neural network determines:

Emotion
Sentiment
Probabilities
Explanation Layer

The LLM does not replace the classifier. Instead, it receives the prediction context and generates a human-readable explanation.

RAG Layer

The RAG system provides grounded project knowledge using:

sentence-transformers/all-MiniLM-L6-v2
384-dimensional embeddings
FAISS IndexFlatIP
persistent metadata
configurable top-k retrieval
Agent Layer

The agent pipeline separates responsibilities:

Interpretation Agent
        ↓
Knowledge Agent
        ↓
Response Agent
        ↓
Safety Agent

The safety layer is designed to prevent unsupported diagnostic claims and keep generated responses aligned with the system's intended scope.

Backend API

The FastAPI service exposes:

Endpoint	Method	Purpose
/health	GET	Service/model health
/predict	POST	Prediction from precomputed modality features
/predict/raw	POST	Raw text/audio/video feature extraction + prediction
/explain	POST	Generate prediction explanation
/rag	POST	Knowledge-grounded retrieval/explanation
/agents	POST	Agent orchestration
/docs	GET	Interactive Swagger API documentation

API documentation:

https://affectra-ai.onrender.com/docs

Frontend

The frontend is built with:

React 18
TypeScript
Vite
Tailwind CSS
React Router
Lucide icons

Main application routes:

/
├── /analysis
├── /explanation
├── /knowledge
├── /agent
└── /about

The UI communicates with the FastAPI backend through a centralized API client and shared prediction context.

Repository Structure
Affectra-AI/
│
├── assets/                         # README and UI assets
│
├── backend/                        # FastAPI application
│   ├── app/
│   │   ├── api/                    # API routes
│   │   ├── core/                   # Configuration
│   │   ├── llm/                    # LLM provider/explanation layer
│   │   ├── schemas/                # Pydantic schemas
│   │   └── services/               # Business/inference services
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # React + Vite application
│   ├── src/
│   │   ├── components/
│   │   ├── context/
│   │   ├── pages/
│   │   └── ...
│   ├── package.json
│   └── vercel.json
│
├── training/                       # Training and inference utilities
│
├── models/                         # Production model metadata/artifacts
│
├── rag/                            # RAG ingestion, embeddings and retrieval
│
├── docs/                           # Architecture and training documentation
│
├── scripts/                        # Validation/deployment/experiment scripts
│
├── tests/                          # Project tests
│
├── render.yaml                     # Render deployment configuration
├── .dockerignore
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
Local Development
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

Install dependencies:

pip install -r backend/requirements.txt

Start FastAPI:

uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

API:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
Frontend
cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173

Configure the backend URL through:

VITE_API_BASE_URL

Do not commit secrets.

Deployment Architecture
                     Internet
                         │
                         ▼
              ┌─────────────────────┐
              │ Vercel              │
              │ React/Vite Frontend │
              └──────────┬──────────┘
                         │ HTTPS
                         ▼
              ┌─────────────────────┐
              │ Render              │
              │ FastAPI Backend     │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Model/RAG Artifacts       API Services
       Hugging Face              Prediction/RAG/
                                 Explanation/Agents

Deployment configuration is included for:

Vercel — frontend
Render — FastAPI backend
Hugging Face — private model/RAG artifact storage
Docker — backend containerization
Production Considerations

The production deployment is intentionally separated into frontend, API, and artifact storage layers.

Large model and RAG artifacts are not committed to the public Git repository. Secrets are supplied through deployment environment variables.

Current deployment limitation

The Render free-tier instance has a 512 MB memory limit. The /predict/raw endpoint can require substantially more memory because it loads heavy pretrained text, audio, and video feature extractors.

Therefore:

the trained fusion model is not the source of the memory issue;
/predict/raw may require a higher-memory inference instance for reliable production use;
this limitation does not require retraining the model;
the repository retains the implementation so the endpoint can be moved to a higher-memory service later.

This limitation is documented explicitly rather than presenting the deployment as fully unrestricted.

Testing and Validation

The project includes validation across multiple layers:

ML
train/dev/test evaluation
per-class precision, recall and F1
confusion analysis
experiment comparison
production artifact verification
Backend
health checks
schema validation
prediction validation
explanation validation
RAG validation
agent validation
Frontend
production build validation
route validation
API integration
responsive UI validation
loading/error states
production API configuration
Deployment
Docker configuration
Render deployment
Vercel deployment
CORS configuration
production API connectivity
Engineering Highlights

This project demonstrates practical ML engineering decisions beyond model training:

Multimodal learning

Three pretrained modality encoders
Common 768-dimensional representation space
Gated multimodal fusion

Model engineering

Frozen pretrained encoders
Compact trainable fusion layer
Separate emotion and sentiment objectives
Exported production inference wrapper

Backend engineering

FastAPI
Pydantic validation
Service-oriented structure
Provider-agnostic LLM layer
Explicit API contracts

Generative AI

RAG with FAISS
Sentence Transformer embeddings
Grounded explanation flow
Specialized agent pipeline
Safety guardrails

Deployment

Docker
Render
Vercel
Hugging Face artifact storage
CORS and environment configuration
Memory-aware lazy loading
Limitations

Affectra AI is an engineering/research project and should not be interpreted as a clinical or psychological diagnostic system.

Current limitations include:

Minority emotion classes have substantially lower F1 than dominant classes.
Emotion recognition remains challenging for ambiguous conversational examples.
Raw multimodal extraction is memory-intensive on low-resource cloud instances.
The LLM is used for explanation and interaction, not as the source of ground-truth emotion labels.
Predictions should be interpreted probabilistically rather than as definitive statements about a person's internal mental state.
Future Improvements

Potential next steps include:

higher-memory inference deployment for raw multimodal processing
optimized/quantized modality encoders
asynchronous media processing
stronger class-imbalance strategies
calibration and uncertainty estimation
multimodal attention visualization
experiment tracking and model registry
automated CI/CD testing
observability and latency monitoring
scalable inference workers
additional conversational datasets
Documentation

Detailed project documentation is available in:

docs/
├── SYSTEM_DESIGN.md
├── TRAINING_ARCHITECTURE.md
└── MIGRATION_PLAN.md

These documents cover architecture, training design, API/deployment decisions, and migration history.

Security and Data Handling

The public repository intentionally excludes:

API keys
access tokens
.env files containing secrets
raw MELD media
large feature caches
private deployment credentials
unnecessary model checkpoints

Use .env.example as the configuration template.

License

This project is licensed under the MIT License. See LICENSE.

Author

Pankaj Yadav

AI/ML Engineering | Multimodal AI | Deep Learning | Generative AI

GitHub: https://github.com/Pankaj429w63
LinkedIn: https://www.linkedin.com/in/pankaj-yadav-0172162a2/
Project Links
Resource	Link
Live Application	https://affectra-ai.vercel.app
GitHub	https://github.com/Pankaj429w63/Affectra-AI
Backend API	https://affectra-ai.onrender.com
Swagger Docs	https://affectra-ai.onrender.com/docs

Affectra AI — More Understanding. A Kinder Tomorrow.
"""

path = Path("/mnt/data/README_Affectra_AI_Advanced.md")
path.write_text(readme, encoding="utf-8")
print(path)

Analyzed
from pathlib import Path
content = """# Affectra AI

> Production-oriented multimodal emotion and sentiment intelligence platform combining text, audio, and video with gated fusion, explainable AI, RAG, and agent orchestration.

**Live Demo:** https://affectra-ai.vercel.app  
**API:** https://affectra-ai.onrender.com  
**API Docs:** https://affectra-ai.onrender.com/docs  
**GitHub:** https://github.com/Pankaj429w63/Affectra-AI

## Overview

Affectra AI is an end-to-end multimodal AI system that combines linguistic, acoustic, and visual information to predict emotion and sentiment.

The system uses pretrained encoders for each modality, maps them into a common 768-dimensional representation, and combines them using a custom Gated Multimodal Fusion network.

**Outputs**
- 7 emotions: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
- 3 sentiments: Positive, Negative, Neutral

The classifier is intentionally separated from the generative AI layer: the ML model produces predictions, while LLM, RAG, and agents provide explanations, grounded knowledge, and interaction.

## Architecture

```text
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
Layer	Technology
Language	Python, TypeScript
ML	PyTorch, Hugging Face Transformers
Text	distilroberta-base
Audio	facebook/wav2vec2-base
Video	google/vit-base-patch16-224
Fusion	Custom Gated Multimodal Fusion
Backend	FastAPI, Pydantic
Frontend	React 18, Vite, Tailwind CSS
RAG	FAISS, SentenceTransformers
LLM	Provider-agnostic explanation layer
Agents	Interpretation, Knowledge, Response, Safety
Deployment	Docker, Render, Vercel
Artifact Storage	Private Hugging Face repository
Dataset

Training and evaluation use the MELD (Multimodal EmotionLines Dataset).

Split	Samples
Train	9,989
Development	1,109
Test	2,610
Total	13,708

Raw datasets, feature caches, large checkpoints, and private artifacts are intentionally excluded from the public repository.

Final Model Results

The selected production experiment is Experiment 2.

Emotion
Metric	Test
Accuracy	55.29%
Weighted Precision	57.36%
Weighted Recall	55.29%
Weighted F1	56.03%
Macro F1	37.01%
Sentiment
Metric	Test
Accuracy	67.16%
Weighted F1	67.01%
Macro F1	64.16%
Emotion F1
Emotion	F1
Anger	36.70%
Disgust	8.64%
Fear	9.72%
Joy	51.21%
Neutral	72.97%
Sadness	29.77%
Surprise	50.08%

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
Endpoint	Method	Purpose
/health	GET	Service and model health
/predict	POST	Prediction from modality features
/predict/raw	POST	Raw media feature extraction + prediction
/explain	POST	Generate prediction explanation
/rag	POST	Knowledge-grounded retrieval
/agents	POST	Agent orchestration
/docs	GET	Swagger documentation

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

.venv\\Scripts\\activate

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
"""
path = Path("/mnt/data/README_Affectra_AI_Advanced.md")
path.write_text(content, encoding="utf-8")
print(f"Created: {path}")
