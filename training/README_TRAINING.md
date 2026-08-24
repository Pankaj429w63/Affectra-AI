# Affectra AI — Training Guide

> **For beginners** — step-by-step instructions to train the multimodal emotion model on Google Colab.

---

## Overview

The model is trained on the **MELD** (Multimodal EmotionLines Dataset) using Google Colab's free GPU.  
The dataset is ~11 GB and is downloaded **directly into Colab storage — never to your local Windows machine**.

Training uses a two-phase approach:
1. **Feature extraction** — run three pretrained encoders once, save outputs to Google Drive
2. **Fusion training** — train only the lightweight Gated Fusion layer (~594K parameters)

---

## Quick Start

### Step 1 — Open the Notebook in Google Colab

**Option A** — Direct link:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste: `https://github.com/Pankaj429w63/Affectra-AI`
4. Select `training/notebooks/AffectraAI_MELD_Training.ipynb`

**Option B** — Upload manually:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `training/notebooks/AffectraAI_MELD_Training.ipynb` from your local machine

---

### Step 2 — Enable GPU

1. In Colab: **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**
4. Verify with Section 02 of the notebook

---

### Step 3 — Run Cells One by One

**Do NOT click "Run all"** — each section has important setup steps and checkpoints.

| Section | Description | Time |
|---|---|---|
| 01 | Mount Google Drive | <1 min |
| 02 | Verify GPU | <1 min |
| 03 | Clone repository | 1–2 min |
| 04 | Install dependencies | 2–4 min |
| 05 | Download MELD (~11 GB) | 10–25 min |
| 06 | Extract + inspect MELD | 3–8 min |
| 07 | Clone annotation repo | 1–2 min |
| 08 | Validate dataset | 2–5 min |
| 09 | Smoke test | 5–10 min |
| ⛔ | **STOP — confirm smoke test passed** | — |
| 10 | Extract all features (text+audio+video) | 2–3 hours |
| 11 | Train fusion model | 20–45 min |
| 12 | Evaluate dev split | 2–5 min |
| 13 | **Final test evaluation (ONCE ONLY)** | 2–5 min |
| 14 | Export model artifacts | 5–10 min |

---

## Where Outputs Are Saved

All outputs are saved to your Google Drive:

```
My Drive/
└── AffectraAI/
    ├── checkpoints/
    │   ├── checkpoint_best.pt       ← best model (based on dev weighted F1)
    │   └── checkpoint_latest.pt     ← most recent checkpoint
    ├── feature_cache/
    │   ├── text_train.pt            ← DistilRoBERTa features, train split
    │   ├── text_dev.pt
    │   ├── text_test.pt
    │   ├── audio_train.pt           ← Wav2Vec2 features, train split
    │   ├── audio_dev.pt
    │   ├── audio_test.pt
    │   ├── video_train.pt           ← ViT features, train split
    │   ├── video_dev.pt
    │   └── video_test.pt
    ├── logs/
    │   └── *.log                    ← training logs per session
    └── training_outputs/
        ├── dataset_validation_report.json
        ├── dev_metrics.json
        ├── test_metrics.json
        └── affectra_multimodal/     ← final model export (backup copy)
            ├── model_state.pt
            ├── model_config.json
            ├── emotion_labels.json
            ├── sentiment_labels.json
            ├── metrics.json
            └── text_encoder/
```

---

## How to Resume After a Session Reset

Colab free tier resets approximately every 12 hours. The pipeline is designed to survive this.

**Steps to resume:**

1. Open the notebook in a new Colab session
2. Run **Section 01** (Mount Drive)  
3. Run **Section 02** (Verify GPU)
4. Run **Section 03** (Clone repo — will do `git pull` if already cloned)
5. Run **Section 04** (Install deps)
6. Skip Sections 05–09 if MELD is already extracted and smoke test passed
7. **Section 10** will automatically skip already-extracted features
8. **Section 11** — change `resume_from=None` to:
   ```python
   resume_from='/content/drive/MyDrive/AffectraAI/checkpoints/checkpoint_latest.pt'
   ```

---

## Out of Memory (OOM) Errors

If you see **"CUDA out of memory"** during training:

1. Open `training/src/config.py` in your repository
2. Find: `BATCH_SIZE: int = 64`
3. Change to: `BATCH_SIZE: int = 32`
4. Commit and push the change
5. In the Colab notebook, re-run Section 03 (`git pull`) to get the updated config
6. Re-run the training cell

If `32` still causes OOM, try `16`.

---

## After Training — Getting Your Model

After Section 14 runs, the model is saved to:

1. **Colab local**: `/content/Affectra-AI/models/affectra_multimodal/`
2. **Google Drive**: `AffectraAI/training_outputs/affectra_multimodal/`

**To use it with the FastAPI backend:**

1. Download `affectra_multimodal/` from the Colab file panel (right sidebar)  
   or download from your Google Drive
2. Place it at: `Affectra-AI/models/affectra_multimodal/`
3. This directory is already git-ignored — do not commit model weights

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No GPU detected` | Runtime → Change runtime type → GPU → Save |
| `wget failed` during MELD download | Check Colab internet, try again — MELD server can be slow |
| `FileNotFoundError: CSV not found` | Re-run Section 06 to confirm MELD extraction succeeded |
| `Import error: training.src.config` | Re-run Section 03 to ensure repo is cloned and in `sys.path` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `config.py` (try 32, then 16) |
| Feature cache shows `❌` | Re-run Section 10 — it will extract only the missing splits |
| Training resumes from wrong epoch | Check `resume_from` path in Section 11 |
| Smoke test crashes | Read the full error, fix the root cause, re-run Section 09 |

---

## Architecture Reference

### Encoders (All Frozen)

| Modality | Model | Output |
|---|---|---|
| Text | `distilroberta-base` | `[B, 768]` via `[CLS]` token |
| Audio | `facebook/wav2vec2-base` | `[B, 768]` via attention-mask mean-pool |
| Video | `google/vit-base-patch16-224` | `[B, 768]` via 8-frame CLS mean-pool |

### Fusion Model (Trainable — ~594K params)

```
Text [768]  → Linear(768→256) + ReLU → Gate → weighted [256]
Audio [768] → Linear(768→256) + ReLU → Gate → weighted [256]
Video [768] → Linear(768→256) + ReLU → Gate → weighted [256]
                     ↓ sum of gated projections
            LayerNorm + Dropout → [256]
              ↙                      ↘
   Linear(256→7)              Linear(256→3)
   Emotion logits              Sentiment logits
```

### Training Config

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping | patience=5, monitor dev emotion weighted F1 |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=3 |
| Loss | Weighted CrossEntropyLoss (handles MELD class imbalance) |
| Emotion loss weight | 0.6 |
| Sentiment loss weight | 0.4 |

---

## MELD Dataset Labels

### Emotions (7 classes)

```
0: anger    1: disgust    2: fear    3: joy
4: neutral  5: sadness    6: surprise
```

### Sentiments (3 classes)

```
0: positive    1: negative    2: neutral
```

---

*See [`docs/TRAINING_ARCHITECTURE.md`](../docs/TRAINING_ARCHITECTURE.md) for full architecture details.*  
*See [`docs/SYSTEM_DESIGN.md`](../docs/SYSTEM_DESIGN.md) for the complete system design.*
