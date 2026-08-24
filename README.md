# Affectra AI — Multimodal Emotion Intelligence Platform

> A production-grade, multimodal AI platform for real-time emotion and sentiment recognition — fusing text, audio, and video signals into a unified inference engine.

---

## Overview

**Affectra AI** is an end-to-end multimodal emotion intelligence system built for research and production use. It combines state-of-the-art deep learning models across three modalities — natural language, speech acoustics, and facial/visual features — to deliver nuanced emotion understanding beyond what single-modality systems can achieve.

The platform is trained on the **MELD** (Multimodal EmotionLines Dataset) benchmark and targets both:

- **7-class emotion recognition** — anger, disgust, fear, joy, neutral, sadness, surprise
- **3-class sentiment analysis** — positive, negative, neutral

---

## Repository Structure

```
Affectra-AI/
├── backend/              # FastAPI inference service
├── frontend/             # React + Vite web application
├── training/
│   ├── src/              # Modular training pipeline (encoders, fusion, trainer)
│   └── notebooks/        # Google Colab training notebooks
├── models/               # Trained artifacts — git-ignored, managed externally
│   └── affectra_multimodal/    # Created after training
│       ├── model_state.pt
│       ├── model_config.json
│       ├── emotion_labels.json
│       ├── sentiment_labels.json
│       ├── metrics.json
│       └── text_encoder/
├── data/                 # Datasets — git-ignored, Colab only, never local
│   └── .gitkeep
├── docs/
│   ├── MIGRATION_PLAN.md
│   ├── SYSTEM_DESIGN.md
│   └── TRAINING_ARCHITECTURE.md
├── scripts/              # Setup and utility scripts
├── .env.example          # Environment variable template (safe placeholders only)
├── .gitignore
├── README.md
└── LICENSE
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **ML Framework** | PyTorch 2.x |
| **Text Encoder** | `distilroberta-base` (Hugging Face) |
| **Audio Encoder** | `facebook/wav2vec2-base` (Hugging Face) |
| **Video Encoder** | `google/vit-base-patch16-224` (Hugging Face) |
| **Fusion Model** | Gated Multimodal Fusion Network (custom, ~594K params) |
| **Backend API** | FastAPI |
| **Frontend** | React + Vite |
| **Training Environment** | Google Colab (free GPU) |
| **Dataset** | MELD (Multimodal EmotionLines Dataset) |
| **Frontend Hosting** | Vercel |
| **Backend Hosting** | Render |
| **Auth + Database** | Supabase |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) | Repository migration from old prototype to new scaffold |
| [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) | Complete system architecture, API contract, deployment topology |
| [`docs/TRAINING_ARCHITECTURE.md`](docs/TRAINING_ARCHITECTURE.md) | ML model design, encoder specs, Colab training workflow |

---

## Getting Started

> **Note:** The application is not yet implemented. This repository is in the architecture/design phase. Implementation will follow iteratively.

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend, when implemented)
- Git
- Google account (for Colab training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Pankaj429w63/Affectra-AI.git
cd Affectra-AI

# Copy env template and fill in your values
cp .env.example .env
# Edit .env with your actual values — NEVER commit this file
```

---

## Training

> The MELD dataset (~11 GB) is **never downloaded locally**. Training runs on Google Colab, which downloads the dataset directly into Colab storage.

Training is conducted in 9 phases via the notebook at `training/notebooks/affectra_train.ipynb`:

1. Validate dataset
2. 100-sample smoke test
3. Extract & cache text features (DistilRoBERTa)
4. Extract & cache audio features (Wav2Vec2)
5. Extract & cache video features (ViT)
6. Train fusion model on cached features
7. Evaluate on dev split
8. Final test evaluation (once only)
9. Export inference artifacts

See [`docs/TRAINING_ARCHITECTURE.md`](docs/TRAINING_ARCHITECTURE.md) for full details.

---

## Environment Variables

Copy `.env.example` to `.env` and populate with your own values. **Never commit `.env`.**

See [`.env.example`](.env.example) for all required variables.

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

---

## Status

| Component | Status |
|---|---|
| Repository scaffold | ✅ Complete |
| System design documentation | ✅ Complete |
| Training architecture documentation | ✅ Complete |
| Training notebook (Colab) | 🔲 Planned |
| Backend API (FastAPI) | 🔲 Planned |
| Frontend UI (React + Vite) | 🔲 Planned |
| Model training on MELD | 🔲 Planned |
