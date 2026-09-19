<div align="center">
  <img src="assets/dashboard.png" alt="Affectra AI Dashboard" width="100%">
  
  <br>
  
  <h1>🧠 Affectra AI</h1>

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

</div>

---

## 🌟 Overview

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

## 🚀 Why This Project Matters

Affectra AI demonstrates a complete ML engineering workflow rather than only a model-training notebook:

**Dataset → Feature Extraction → Multimodal Fusion → Evaluation → Model Export → API → RAG → Agents → Frontend → Cloud Deployment**

The project covers practical engineering concerns including:
- Pretrained Transformer integration
- Multimodal representation learning
- Gated feature fusion
- Multi-task classification
- Reproducible evaluation
- API design and validation
- Retrieval-augmented generation (RAG)
- Agent orchestration
- Frontend/backend integration
- Docker & cloud deployment
- Memory-aware inference
- Production configuration and CORS

---

## 🏗️ System Architecture

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
```

---

## 🧠 Machine Learning Pipeline

### 1. Pretrained Encoders

| Modality | Encoder | Output Size |
| :--- | :--- | :--- |
| **Text** | `distilroberta-base` | 768 |
| **Audio** | `facebook/wav2vec2-base` | 768 |
| **Video** | `google/vit-base-patch16-224` | 768 |

The encoders are used as frozen feature extractors during fusion training.

### 2. Feature Fusion

The three modality representations are passed to a custom Gated Multimodal Fusion network.

```text
Text   ──► 768 ──┐
Audio  ──► 768 ──┼──► Gated Fusion ──► 512
Video  ──► 768 ──┘
                         │
                         ├──► Emotion classifier (7)
                         └──► Sentiment classifier (3)
```

The fusion model contains approximately **594K trainable parameters**, making the trainable component substantially smaller than the pretrained encoders.

### 3. Prediction
The prediction system returns probability distributions rather than only a hard label, allowing the application layer to expose confidence information and supporting downstream explanation.

---

## 📊 Dataset & Model Results

The multimodal training pipeline is based on the **MELD (Multimodal EmotionLines Dataset)**.

| Split | Samples |
| :--- | :--- |
| **Train** | 9,989 |
| **Development** | 1,109 |
| **Test** | 2,610 |
| **Total** | **13,708** |

*Note: Raw media, feature caches, and large checkpoints are intentionally excluded from the public Git repository.*

### Final Model Results (Experiment 2)

#### Emotion (7-class)
| Metric | Test Score |
| :--- | :--- |
| Accuracy | 55.29% |
| Weighted Precision | 57.36% |
| Weighted Recall | 55.29% |
| Weighted F1 | 56.03% |
| Macro F1 | 37.01% |

#### Sentiment (3-class)
| Metric | Test Score |
| :--- | :--- |
| Accuracy | 67.16% |
| Weighted F1 | 67.01% |
| Macro F1 | 64.16% |

#### Emotion F1 by Class
| Emotion | F1 Score |
| :--- | :--- |
| Anger | 36.70% |
| Disgust | 8.64% |
| Fear | 9.72% |
| Joy | 51.21% |
| Neutral | 72.97% |
| Sadness | 29.77% |
| Surprise | 50.08% |

> The macro/weighted metric difference reflects the class imbalance and the difficulty of learning minority emotion classes. This is an important limitation and is intentionally reported rather than hidden.

---

## 🔍 Explainability and Grounded AI

Affectra separates prediction from generation.

### Prediction Layer
The multimodal neural network determines the **Emotion**, **Sentiment**, and their **Probabilities**.

### Explanation Layer
The LLM does not replace the classifier. Instead, it receives the prediction context and generates a human-readable explanation.

### RAG Layer
The RAG system provides grounded project knowledge using:
- `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- FAISS IndexFlatIP
- Configurable top-k retrieval

### Agent Layer
The agent pipeline separates responsibilities:
```text
Interpretation Agent  ➔  Knowledge Agent  ➔  Response Agent  ➔  Safety Agent
```
The safety layer is designed to prevent unsupported diagnostic claims and keep generated responses aligned with the system's intended scope.

---

## 🔌 Backend API

The FastAPI service exposes the following endpoints:

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | GET | Service/model health |
| `/predict` | POST | Prediction from precomputed modality features |
| `/predict/raw` | POST | Raw text/audio/video feature extraction + prediction |
| `/explain` | POST | Generate prediction explanation |
| `/rag` | POST | Knowledge-grounded retrieval/explanation |
| `/agents` | POST | Agent orchestration |
| `/docs` | GET | Interactive Swagger API documentation |

---

## 💻 Frontend Dashboard

The frontend is built with **React 18, TypeScript, Vite, and Tailwind CSS**.

Main application routes:
```text
/
├── /analysis
├── /explanation
├── /knowledge
├── /agent
└── /about
```

---

## 📁 Repository Structure

```text
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
```

---

## ⚙️ Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone
```bash
git clone https://github.com/Pankaj429w63/Affectra-AI.git
cd Affectra-AI
```

### 2. Backend
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI available at: `http://127.0.0.1:8000/docs`

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend available at: `http://localhost:5173`

*(Configure the backend URL through `VITE_API_BASE_URL` in your environment files.)*

---

## 🐳 Deployment Architecture

```text
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
```

### ⚠️ Production Limitation (Render Free Tier)
The current Render free instance has a **512 MB memory limit**. The `/predict/raw` endpoint can exceed this limit because raw multimodal processing loads heavy pretrained text, audio, and video encoders.

This is a deployment-resource limitation, not a training failure. The trained fusion model remains unchanged. Reliable raw multimodal production inference requires a higher-memory inference instance or an optimized extraction service.

---

## 🛡️ Security and Data Handling

The public repository intentionally excludes:
- API keys and access tokens
- `.env` files containing secrets
- Raw MELD media datasets
- Large feature caches and checkpoints

Secrets are supplied strictly through deployment environment variables.

---

## 📄 License & Author

This project is licensed under the MIT License. See `LICENSE`.

**Author:** Pankaj Yadav  
*AI/ML Engineering | Multimodal AI | Deep Learning | Generative AI*  
- **GitHub:** [https://github.com/Pankaj429w63](https://github.com/Pankaj429w63)
- **LinkedIn:** [Pankaj Yadav](https://www.linkedin.com/in/pankaj-yadav-0172162a2/)

<br>

<div align="center">
  <i>"More Understanding. A Kinder Tomorrow."</i>
</div>
