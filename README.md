<div align="center">
  <img src="assets/dashboard.png" alt="Affectra AI Dashboard" width="100%">
  
  <br>
  
  <h1>🧠 Affectra AI</h1>
  
  <p><strong>A Production-Grade Multimodal Emotion Intelligence Platform</strong></p>
  
  <p>
    <a href="https://github.com/Pankaj429w63/Affectra-AI/stargazers"><img src="https://img.shields.io/github/stars/Pankaj429w63/Affectra-AI?style=for-the-badge&color=blue" alt="Stars"></a>
    <a href="https://github.com/Pankaj429w63/Affectra-AI/network/members"><img src="https://img.shields.io/github/forks/Pankaj429w63/Affectra-AI?style=for-the-badge&color=blue" alt="Forks"></a>
    <a href="https://github.com/Pankaj429w63/Affectra-AI/issues"><img src="https://img.shields.io/github/issues/Pankaj429w63/Affectra-AI?style=for-the-badge&color=blue" alt="Issues"></a>
    <a href="https://github.com/Pankaj429w63/Affectra-AI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Pankaj429w63/Affectra-AI?style=for-the-badge&color=blue" alt="License"></a>
  </p>
</div>

---

## 🌟 Overview

**Affectra AI** is a state-of-the-art, end-to-end multimodal emotion intelligence system built for both research and production. It goes beyond simple text analysis by fusing **natural language, speech acoustics, and visual facial features** into a unified inference engine to achieve nuanced, human-level emotion and sentiment understanding.

Trained on the prestigious **MELD (Multimodal EmotionLines Dataset)** benchmark, Affectra AI precisely identifies:
- 🎭 **7 Emotion Classes:** Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
- ⚖️ **3 Sentiment Classes:** Positive, Negative, Neutral

---

## ✨ Key Features

- **🗣️ Multimodal Fusion:** Dynamically learns and weights the importance of text, audio, and video inputs using a custom Gated Multimodal Fusion Network.
- **⚡ Real-Time Inference:** Blazing fast FastAPI backend capable of processing raw media files directly into embeddings and predictions.
- **🎨 Beautiful UI:** A highly polished, responsive React + Vite web dashboard featuring interactive charts and glassmorphic micro-animations.
- **🧠 Explainable AI & RAG:** Integrated Retrieval-Augmented Generation (RAG) and Agentic pipelines to transparently explain *why* the model made a prediction based on knowledge bases.
- **🔒 Production Ready:** Fully containerized with Docker, complete with Vercel and Render deployment configurations.

---

## 🛠️ Technology Stack

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React">
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</div>

| Layer | Technology / Model |
|---|---|
| **Text Encoder** | `distilroberta-base` (Hugging Face) |
| **Audio Encoder** | `facebook/wav2vec2-base` (Hugging Face) |
| **Video Encoder** | `google/vit-base-patch16-224` (Hugging Face) |
| **Fusion Model** | Custom Gated Multimodal Fusion Network (~594K params) |
| **Backend API** | FastAPI (Python 3.11+) |
| **Frontend** | React 18 + Vite + TailwindCSS |
| **RAG System** | FAISS + SentenceTransformers |

---

## 🏗️ Repository Structure

```text
Affectra-AI/
├── backend/              # FastAPI inference service & endpoints
├── frontend/             # React + Vite web application (Dashboard)
├── training/             # Modular training pipeline (encoders, fusion, trainer)
├── models/               # Trained artifacts & model checkpoints
├── rag/                  # FAISS vectorstore and retriever logic
├── docs/                 # System architecture and design documentation
├── assets/               # README assets and images
├── Dockerfile            # (backend) Production container definition
├── render.yaml           # Render deployment configuration
└── .env.example          # Environment variable template
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- Git

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Pankaj429w63/Affectra-AI.git
cd Affectra-AI

# Create environment file
cp .env.example .env
```

### 2. Start the Backend (FastAPI)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the server (runs on http://localhost:8000)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the Frontend (React + Vite)
```bash
cd frontend

# Install dependencies
npm install

# Start development server (runs on http://localhost:5173)
npm run dev
```

---

## 📚 Documentation

For deep dives into the architecture, model design, and deployment strategies, refer to our comprehensive documentation:

| Document | Description |
|---|---|
| [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Complete system architecture, API contract, and deployment topology. |
| [`docs/TRAINING_ARCHITECTURE.md`](docs/TRAINING_ARCHITECTURE.md) | ML model design, encoder specs, and Google Colab training workflow. |
| [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) | Legacy migration notes from early prototypes. |

---

## 🐳 Deployment

Affectra AI is fully configured for modern cloud deployment:
- **Backend (Render):** A complete Dockerfile and `render.yaml` are provided in the repository root for one-click deployment to Render.
- **Frontend (Vercel):** The React SPA is pre-configured with a `vercel.json` routing configuration for seamless Vercel hosting.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br>

<div align="center">
  <i>"More Understanding. A Kinder Tomorrow."</i>
</div>
