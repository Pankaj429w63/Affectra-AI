# Affectra AI — System Design

> **Version:** 1.0  
> **Date:** 2026-08-24  
> **Status:** Architecture definition — implementation not yet started  
> **Author:** Pankaj Yadav

---

## 1. Project Overview

**Affectra AI** is a multimodal emotion intelligence platform that fuses text, audio, and video signals to predict human emotion and sentiment in real time.

| Property | Value |
|---|---|
| Task | Multimodal emotion + sentiment classification |
| Dataset | MELD (Multimodal EmotionLines Dataset) |
| Emotion classes | 7 — anger, disgust, fear, joy, neutral, sadness, surprise |
| Sentiment classes | 3 — positive, negative, neutral |
| Backend | FastAPI (Python) |
| Frontend | React + Vite |
| Training environment | Google Colab (free GPU tier) |
| Deployment (frontend) | Vercel |
| Deployment (backend) | Render |

---

## 2. Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                    (React + Vite Frontend)                       │
│                                                                 │
│   ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│   │  Text    │  │  Audio File  │  │    Video File / URL    │   │
│   │  Input   │  │  Upload .wav │  │    Upload .mp4         │   │
│   └────┬─────┘  └──────┬───────┘  └──────────┬────────────┘   │
│        └───────────────┴─────────────────────┘                 │
│                         │  multipart/form-data                  │
└─────────────────────────┼───────────────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI BACKEND (Render)                      │
│                                                                 │
│   POST /api/v1/predict                                          │
│   ┌──────────────────────────────────────────────────────┐     │
│   │  Request Router & Validator                          │     │
│   │  • Validates input fields (text / audio / video)    │     │
│   │  • Returns 422 if all inputs are missing            │     │
│   └───────────────┬──────────────────────────────────────┘     │
│                   │                                             │
│   ┌───────────────▼──────────────────────────────────────┐     │
│   │  Inference Engine                                    │     │
│   │                                                      │     │
│   │  Text?  ──► Text Feature Extractor (DistilRoBERTa)  │     │
│   │  Audio? ──► Audio Feature Extractor (Wav2Vec2)      │     │
│   │  Video? ──► Video Feature Extractor (ViT-Base)      │     │
│   │                                                      │     │
│   │  Available features ──► Gated Multimodal Fusion     │     │
│   │                                                      │     │
│   │  Fusion Output ──► Emotion Head (7 classes)         │     │
│   │                └──► Sentiment Head (3 classes)       │     │
│   └───────────────┬──────────────────────────────────────┘     │
│                   │                                             │
│   ┌───────────────▼──────────────────────────────────────┐     │
│   │  JSON Response                                       │     │
│   │  {                                                   │     │
│   │    "emotion": "joy",                                 │     │
│   │    "emotion_confidence": 0.87,                       │     │
│   │    "emotion_probs": { ... },                         │     │
│   │    "sentiment": "positive",                          │     │
│   │    "sentiment_confidence": 0.91,                     │     │
│   │    "sentiment_probs": { ... },                       │     │
│   │    "modalities_used": ["text", "audio"]              │     │
│   │  }                                                   │     │
│   └──────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Frontend → Backend → Model Flow

```
Frontend (Vite + React)
        │
        │  1. User fills text input and/or uploads audio/video
        │  2. Form submitted via fetch() / axios POST
        │  3. multipart/form-data payload
        │
        ▼
FastAPI Backend
        │
        │  4. Route: POST /api/v1/predict
        │  5. Validate: at least one modality present
        │  6. Pre-process each modality present:
        │       text  → tokenize → encode → [1, 768]
        │       audio → resample to 16kHz → encode → [1, 768]
        │       video → decode frames → resize 224x224 → encode → [1, 768]
        │  7. Pass available feature vectors to fusion model
        │  8. Gated fusion: learn to weight modalities dynamically
        │  9. Emotion head → softmax over 7 classes
        │ 10. Sentiment head → softmax over 3 classes
        │
        ▼
Affectra Fusion Model
(loaded once at startup from models/affectra_multimodal/)
        │
        │ 11. Returns (emotion_logits, sentiment_logits)
        │
        ▼
FastAPI Backend
        │
        │ 12. Build JSON response with labels + probabilities
        │ 13. Return HTTP 200 with prediction result
        │
        ▼
Frontend
        │
        │ 14. Render emotion label, confidence bar, all class probabilities
        │ 15. Show which modalities were used
```

---

## 4. Inference Flow Detail (Single Request Lifecycle)

```
POST /api/v1/predict
├── Input validation
│   ├── text: Optional[str]
│   ├── audio: Optional[UploadFile]  (.wav, .mp3)
│   └── video: Optional[UploadFile]  (.mp4, .avi)
│
├── Feature extraction (only for provided modalities)
│   ├── Text branch
│   │   ├── Tokenize with DistilRoBERTa tokenizer
│   │   ├── Encode with frozen DistilRoBERTa
│   │   └── Extract [CLS] token → shape [1, 768]
│   │
│   ├── Audio branch
│   │   ├── Load audio file → resample to 16 kHz mono
│   │   ├── Encode with frozen Wav2Vec2-base
│   │   └── Mean-pool time dimension → shape [1, 768]
│   │
│   └── Video branch
│       ├── Decode video → sample N frames (e.g., 8)
│       ├── Resize each frame to 224×224, normalize
│       ├── Encode each frame with frozen ViT-base
│       └── Mean-pool across frames → shape [1, 768]
│
├── Gated multimodal fusion
│   ├── Project each available feature: [1, 768] → [1, 256]
│   ├── Learn scalar gate weights per modality
│   ├── Weighted sum of projected features
│   └── Output fused vector: [1, 256]
│
├── Task heads
│   ├── Emotion head: Linear(256, 7) → softmax
│   └── Sentiment head: Linear(256, 3) → softmax
│
└── Response
    ├── predicted_emotion: str
    ├── emotion_confidence: float
    ├── emotion_probs: dict[str, float]  (all 7 classes)
    ├── predicted_sentiment: str
    ├── sentiment_confidence: float
    ├── sentiment_probs: dict[str, float]  (all 3 classes)
    └── modalities_used: list[str]
```

---

## 5. Final Folder Structure

```
Affectra-AI/
│
├── backend/                          # FastAPI inference service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── config.py                 # Settings via pydantic-settings
│   │   ├── routers/
│   │   │   └── predict.py            # POST /api/v1/predict
│   │   ├── services/
│   │   │   ├── inference.py          # Loads model, runs prediction
│   │   │   ├── text_encoder.py       # DistilRoBERTa feature extraction
│   │   │   ├── audio_encoder.py      # Wav2Vec2 feature extraction
│   │   │   └── video_encoder.py      # ViT feature extraction
│   │   └── models/
│   │       └── schemas.py            # Pydantic request/response schemas
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                         # React + Vite web application
│   ├── public/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── InputPanel.jsx
│   │   │   ├── ResultPanel.jsx
│   │   │   └── EmotionBar.jsx
│   │   ├── api/
│   │   │   └── predict.js            # fetch() wrapper for /api/v1/predict
│   │   └── styles/
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── training/                         # Colab-based training pipeline
│   ├── src/
│   │   ├── dataset.py                # MELD dataset loader
│   │   ├── encoders.py               # Pretrained encoder wrappers
│   │   ├── feature_cache.py          # Save/load cached features
│   │   ├── fusion_model.py           # Gated multimodal fusion + heads
│   │   ├── trainer.py                # Training loop
│   │   ├── evaluator.py              # Evaluation + metrics
│   │   └── export.py                 # Save artifacts to models/
│   └── notebooks/
│       └── affectra_train.ipynb      # Main Colab training notebook
│
├── models/                           # Trained artifacts (git-ignored)
│   ├── .gitkeep
│   └── affectra_multimodal/          # Created after training
│       ├── model_state.pt            # Fusion model weights only
│       ├── model_config.json         # Architecture hyperparameters
│       ├── emotion_labels.json       # 7-class label mapping
│       ├── sentiment_labels.json     # 3-class label mapping
│       ├── metrics.json              # Final evaluation metrics
│       └── text_encoder/             # Saved tokenizer for inference
│
├── data/                             # Datasets (git-ignored, Colab-only)
│   └── .gitkeep
│
├── docs/
│   ├── MIGRATION_PLAN.md
│   ├── SYSTEM_DESIGN.md              # This file
│   └── TRAINING_ARCHITECTURE.md
│
├── scripts/
│   └── .gitkeep
│
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## 6. Deployment Architecture

### 6.1 Production Topology

```
┌──────────────────┐     HTTPS      ┌─────────────────────────┐
│  Vercel          │ ◄────────────► │  Render                  │
│  (React + Vite)  │                │  (FastAPI)               │
│  affectra.ai     │                │  api.affectra.ai         │
└──────────────────┘                └─────────────┬───────────┘
                                                  │
                                    ┌─────────────▼───────────┐
                                    │  Supabase               │
                                    │  (Auth + Postgres DB)   │
                                    └─────────────────────────┘
```

### 6.2 Services Summary

| Service | Provider | Purpose | Cost |
|---|---|---|---|
| Frontend hosting | Vercel | React + Vite build & CDN | Free tier |
| Backend hosting | Render | FastAPI Docker container | Free tier |
| Auth + Database | Supabase | User auth, request logs | Free tier |
| Model artifacts | Git-ignored, copied to Render on deploy | `models/affectra_multimodal/` | — |

---

## 7. Future GenAI / RAG / Agentic Expansion

> These components are **not part of the initial implementation**. They will be added only after the core multimodal model is working and evaluated.

```
Current (v1)                     Future (v2+)
────────────────────────────     ─────────────────────────────────────
                                 ┌──────────────────────────────────┐
                                 │  Agentic Layer                   │
                                 │  • LangGraph emotion agent       │
                                 │  • Multi-turn conversation       │
                                 │  • Tool use (calendar, alerts)   │
                                 └────────────────┬─────────────────┘
                                                  │
User ──► FastAPI ──► Fusion ──► Output            │
                                 ┌────────────────▼─────────────────┐
                                 │  RAG Layer                       │
                                 │  • FAISS vector store            │
                                 │  • Emotion-context retrieval     │
                                 │  • Contextual response gen       │
                                 └────────────────┬─────────────────┘
                                                  │
                                 ┌────────────────▼─────────────────┐
                                 │  Local GenAI (Ollama)            │
                                 │  • llama3 / mistral              │
                                 │  • Private, offline inference    │
                                 │  • Developer experimentation     │
                                 └──────────────────────────────────┘
```

### 7.1 Expansion Phases

| Phase | Feature | Technology |
|---|---|---|
| v1 (current) | Multimodal emotion/sentiment classification | PyTorch, FastAPI, React |
| v2 | Conversation history + session tracking | Supabase Postgres |
| v3 | RAG: emotion-aware contextual responses | FAISS, LangChain |
| v4 | Agentic workflows | LangGraph |
| v5 | Local LLM integration | Ollama (llama3/mistral) |

---

## 8. API Contract (Initial Version)

### `POST /api/v1/predict`

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | No* | Natural language utterance |
| `audio` | file | No* | WAV or MP3 audio file |
| `video` | file | No* | MP4 or AVI video file |

*At least one field must be provided.

**Response** — `application/json`

```json
{
  "emotion": "joy",
  "emotion_confidence": 0.87,
  "emotion_probs": {
    "anger": 0.02,
    "disgust": 0.01,
    "fear": 0.01,
    "joy": 0.87,
    "neutral": 0.05,
    "sadness": 0.02,
    "surprise": 0.02
  },
  "sentiment": "positive",
  "sentiment_confidence": 0.91,
  "sentiment_probs": {
    "positive": 0.91,
    "negative": 0.04,
    "neutral": 0.05
  },
  "modalities_used": ["text", "audio"]
}
```

**Error Responses**

| Code | Meaning |
|---|---|
| `422 Unprocessable Entity` | No modality provided |
| `415 Unsupported Media Type` | Unsupported file format |
| `500 Internal Server Error` | Model inference failure |

---

## 9. Environment Variables Reference

See [`.env.example`](../.env.example) for the full list. Key variables:

| Variable | Used By | Description |
|---|---|---|
| `MODEL_DIR` | Backend | Path to `models/affectra_multimodal/` |
| `API_HOST` | Backend | Bind host (default `0.0.0.0`) |
| `API_PORT` | Backend | Bind port (default `8000`) |
| `API_SECRET_KEY` | Backend | Auth secret (change in production) |
| `VITE_API_URL` | Frontend | Backend URL for API calls |
| `SUPABASE_URL` | Backend | Supabase project URL |
| `SUPABASE_ANON_KEY` | Backend | Supabase public anon key |

---

## 10. Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **FastAPI over Flask** | Async support, automatic OpenAPI docs, Pydantic validation, production-ready |
| **React + Vite over Next.js** | Simpler for a beginner SPA; no SSR complexity needed at this stage |
| **DistilRoBERTa over full RoBERTa** | 40% smaller, 60% faster, ~97% of RoBERTa accuracy — fits free Colab GPU |
| **Wav2Vec2-base over Wav2Vec2-large** | 95M params vs 317M — feasible on free Colab T4 |
| **ViT-base over ResNet** | Stronger video frame representation; pre-trained on ImageNet-21k |
| **Feature caching** | Freeze encoders after extraction; only train lightweight fusion layer |
| **Gated fusion over concatenation** | Learns to down-weight missing/noisy modalities automatically |
| **Official MELD splits** | Preserves dialogue-level conversation structure; prevents data leakage |
| **Vercel + Render + Supabase** | Entirely free tier for initial deployment |
| **FAISS for RAG** | Simple, local, no external service needed; swap to Pinecone later |

---

*See [`TRAINING_ARCHITECTURE.md`](TRAINING_ARCHITECTURE.md) for the detailed model design and Colab training workflow.*
