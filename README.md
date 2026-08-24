# Affectra AI — Multimodal Emotion Intelligence Platform

> A production-grade, multimodal AI platform for real-time emotion and sentiment recognition — fusing text, audio, and video signals into a unified inference engine.

---

## Overview

**Affectra AI** is an end-to-end multimodal emotion intelligence system built for research and production use. It combines state-of-the-art deep learning models across three modalities — natural language, speech acoustics, and facial/visual features — to deliver nuanced emotion understanding beyond what single-modality systems can achieve.

The platform is designed around the **MELD** (Multimodal EmotionLines Dataset) benchmark and targets both:

- **7-class emotion recognition** — Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
- **3-class sentiment analysis** — Positive, Negative, Neutral

---

## Repository Structure

```
Affectra-AI/
├── backend/              # Inference API service (FastAPI / Flask)
├── frontend/             # Web application UI
├── training/
│   ├── src/              # Modular training pipeline (models, data, trainers)
│   └── notebooks/        # EDA, experiments, and visualisation notebooks
├── models/               # Model checkpoints — git-ignored, managed externally
│   └── .gitkeep
├── data/                 # Datasets — git-ignored, never committed
│   └── .gitkeep
├── docs/                 # Architecture docs and design notes
│   └── MIGRATION_PLAN.md
├── scripts/              # Setup, download, and export utilities
├── .env.example          # Environment variable template (safe placeholders only)
├── .gitignore
├── README.md
└── LICENSE
```

---

## Tech Stack (Planned)

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **ML Framework** | PyTorch 2.x |
| **Text Encoder** | BERT / RoBERTa (Hugging Face Transformers) |
| **Audio Encoder** | Wav2Vec 2.0 / MFCC |
| **Video Encoder** | ResNet / FaceNet |
| **Fusion** | Cross-modal attention + LSTM context |
| **Backend API** | FastAPI |
| **Frontend** | React / Next.js |
| **Experiment Tracking** | Weights & Biases / MLflow |
| **Dataset** | MELD (Multimodal EmotionLines Dataset) |

---

## Getting Started

> **Note:** The application is not yet implemented. This repository is currently in the scaffold phase. Implementation will follow iteratively.

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Pankaj429w63/Affectra-AI.git
cd Affectra-AI

# Copy env template and fill in your values
cp .env.example .env
```

---

## Environment Variables

Copy `.env.example` to `.env` and populate with your own values. **Never commit `.env`.**

See [`.env.example`](.env.example) for all required variables.

---

## Documentation

- [`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md) — Repository migration and cleanup plan

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

---

## Status

| Component | Status |
|---|---|
| Repository scaffold | ✅ Complete |
| Training pipeline | 🔲 Planned |
| Backend API | 🔲 Planned |
| Frontend UI | 🔲 Planned |
| Model training (MELD) | 🔲 Planned |
