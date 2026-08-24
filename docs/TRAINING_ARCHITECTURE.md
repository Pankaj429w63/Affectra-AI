# Affectra AI — Training Architecture

> **Version:** 1.0  
> **Date:** 2026-08-24  
> **Status:** Architecture definition — implementation not yet started  
> **Training environment:** Google Colab (free GPU, T4 / L4)

---

## 1. Dataset: MELD (Multimodal EmotionLines Dataset)

### 1.1 Overview

| Property | Value |
|---|---|
| Full name | Multimodal EmotionLines Dataset |
| Source | Friends TV show dialogues |
| Modalities | Text (transcripts), Audio (`.mp4` → WAV), Video (`.mp4` frames) |
| Total size | ~11 GB |
| Storage | **Colab runtime only — never downloaded locally** |

### 1.2 Official Splits (Always Use These — Never Random-Split)

| Split | Dialogues | Utterances | Purpose |
|---|---|---|---|
| `train` | 1,039 | 9,989 | Model training |
| `dev` | 114 | 1,109 | Hyperparameter tuning, early stopping |
| `test` | 280 | 2,610 | Final one-time evaluation only |

> **Rule:** The test split is touched **exactly once** — at the end of training (Phase 8). Evaluate on `dev` during development.

### 1.3 Official Emotion Labels

```python
EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}
```

### 1.4 Official Sentiment Labels

```python
SENTIMENT_LABELS = {
    0: "positive",
    1: "negative",
    2: "neutral"
}
```

### 1.5 Class Distribution (MELD Train Set — Approximate)

| Emotion | Count | % |
|---|---|---|
| neutral | 4,710 | 47.1% |
| surprise | 1,205 | 12.1% |
| fear | 268 | 2.7% |
| sadness | 683 | 6.8% |
| joy | 1,743 | 17.4% |
| disgust | 271 | 2.7% |
| anger | 1,109 | 11.1% |

> **Note:** MELD is heavily imbalanced. Use weighted cross-entropy loss during training.

### 1.6 Dataset Download in Colab

```python
# In Colab notebook — download directly into Colab, not your local machine
import os

MELD_URL = "https://affective-meld.github.io/resources/MELD.Raw.tar.gz"
COLAB_DATA_DIR = "/content/meld_data"

os.makedirs(COLAB_DATA_DIR, exist_ok=True)

# Download
!wget -O /content/MELD.Raw.tar.gz {MELD_URL}

# Extract
!tar -xzf /content/MELD.Raw.tar.gz -C {COLAB_DATA_DIR}

# Annotation CSVs will be at:
# /content/meld_data/MELD.Raw/train_sent_emo.csv
# /content/meld_data/MELD.Raw/dev_sent_emo.csv
# /content/meld_data/MELD.Raw/test_sent_emo.csv
```

---

## 2. Model Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                 │
│  ┌────────────┐   ┌─────────────────┐   ┌──────────────────┐  │
│  │  Text      │   │  Audio          │   │  Video           │  │
│  │  (string)  │   │  (wav, 16kHz)   │   │  (mp4 frames)    │  │
│  └─────┬──────┘   └────────┬────────┘   └────────┬─────────┘  │
└────────┼──────────────────┼────────────────────┼──────────────┘
         │                  │                    │
┌────────▼──────────────────▼────────────────────▼──────────────┐
│                  FROZEN PRETRAINED ENCODERS                     │
│  ┌────────────────────┐ ┌──────────────────┐ ┌──────────────┐ │
│  │  DistilRoBERTa     │ │  Wav2Vec2-base   │ │  ViT-base    │ │
│  │  (distilroberta-   │ │  (facebook/      │ │  (google/    │ │
│  │   base)            │ │  wav2vec2-base)  │ │  vit-base-   │ │
│  │                    │ │                  │ │  patch16-    │ │
│  │  Tokenize → encode │ │  Resample 16kHz  │ │  224)        │ │
│  │  CLS token pool    │ │  → encode        │ │              │ │
│  │                    │ │  Mean-pool time  │ │  N frames    │ │
│  │  Output: [1, 768]  │ │  Output: [1,768] │ │  → encode    │ │
│  │                    │ │                  │ │  Mean-pool   │ │
│  │                    │ │                  │ │  [1, 768]    │ │
│  └────────┬───────────┘ └────────┬─────────┘ └──────┬───────┘ │
└───────────┼─────────────────────┼────────────────────┼─────────┘
            │   (cached to disk)  │  (cached to disk)  │
┌───────────▼─────────────────────▼────────────────────▼─────────┐
│              FEATURE CACHE LAYER (Colab Drive)                  │
│  text_features_{split}.pt   audio_features_{split}.pt           │
│  video_features_{split}.pt  (shape: [N_utterances, 768])        │
└───────────┬─────────────────────┬────────────────────┬─────────┘
            │                     │                    │
┌───────────▼─────────────────────▼────────────────────▼─────────┐
│              GATED MULTIMODAL FUSION MODEL (Trainable)          │
│                                                                 │
│  For each modality present:                                     │
│  Linear(768, 256) + ReLU  →  projected feature [1, 256]        │
│                                                                 │
│  Gate network:                                                  │
│  gate_i = sigmoid(Linear(256, 1))  →  scalar weight [0,1]      │
│                                                                 │
│  Fused = Σ (gate_i × projected_i)  →  [1, 256]                 │
│                                                                 │
│  Dropout(0.3) → LayerNorm(256)                                  │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────┐       │
│  │  Emotion Head       │   │  Sentiment Head            │       │
│  │  Linear(256, 7)     │   │  Linear(256, 3)            │       │
│  │  Softmax            │   │  Softmax                   │       │
│  │  → 7 class probs    │   │  → 3 class probs           │       │
│  └─────────────────────┘   └───────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Encoder Details

### 3.1 Text Encoder — DistilRoBERTa

| Property | Value |
|---|---|
| Model ID | `distilroberta-base` |
| Parameters | ~82M |
| Input | Raw text string |
| Tokenizer | `AutoTokenizer.from_pretrained("distilroberta-base")` |
| Max token length | 128 (MELD utterances are short) |
| Pooling strategy | `[CLS]` token hidden state |
| Output shape | `[batch, 768]` |
| Frozen during fusion training | ✅ Yes |
| Why DistilRoBERTa | 40% smaller than RoBERTa-base; ideal for Colab free tier |

```python
# Example extraction
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModel.from_pretrained("distilroberta-base")
model.eval()

inputs = tokenizer("I'm so happy!", return_tensors="pt",
                   truncation=True, max_length=128, padding="max_length")
with torch.no_grad():
    outputs = model(**inputs)
text_feature = outputs.last_hidden_state[:, 0, :]  # [CLS] → [1, 768]
```

### 3.2 Audio Encoder — Wav2Vec2-base

| Property | Value |
|---|---|
| Model ID | `facebook/wav2vec2-base` |
| Parameters | ~95M |
| Input | Raw 16 kHz mono waveform |
| Processor | `Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")` |
| Pooling strategy | Mean-pool across time frames |
| Output shape | `[batch, 768]` |
| Frozen during fusion training | ✅ Yes |
| Audio preprocessing | Resample to 16 kHz, convert to mono, normalize |
| Why Wav2Vec2-base | Strong speech representation; fits Colab free T4 |

```python
# Example extraction
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

waveform, sr = librosa.load("utterance.wav", sr=16000, mono=True)
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt",
                   padding=True)
with torch.no_grad():
    outputs = model(**inputs)
audio_feature = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
```

### 3.3 Video Encoder — ViT-base-patch16-224

| Property | Value |
|---|---|
| Model ID | `google/vit-base-patch16-224` |
| Parameters | ~86M |
| Input | RGB frames resized to 224×224 |
| Processor | `ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")` |
| Frames sampled per clip | 8 (uniformly sampled) |
| Pooling strategy | Mean-pool `[CLS]` tokens across all 8 frames |
| Output shape | `[batch, 768]` |
| Frozen during fusion training | ✅ Yes |
| Why ViT-base | Better visual representation than ResNet; ImageNet-21k pretraining |

```python
# Example extraction
from transformers import ViTFeatureExtractor, ViTModel
import torch
import cv2
import numpy as np

extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.eval()

# Sample 8 frames from video
cap = cv2.VideoCapture("utterance.mp4")
frames = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total_frames - 1, num=8, dtype=int)
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
cap.release()

inputs = extractor(images=frames, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# CLS token for each of 8 frames, then mean-pool
video_feature = outputs.last_hidden_state[:, 0, :].mean(dim=0, keepdim=True)  # [1, 768]
```

---

## 4. Fusion Model Architecture

### 4.1 GatedMultimodalFusion

```python
class GatedMultimodalFusion(nn.Module):
    """
    Learns a scalar gate weight per modality.
    Missing modalities are masked out (zero vector input → zero gate output).
    Only the lightweight fusion layer is trained; encoders are frozen.
    """
    def __init__(
        self,
        input_dim: int = 768,       # All three encoders output 768-dim
        fusion_dim: int = 256,       # Projected dimension
        num_emotions: int = 7,
        num_sentiments: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # One projection per modality
        self.text_proj    = nn.Linear(input_dim, fusion_dim)
        self.audio_proj   = nn.Linear(input_dim, fusion_dim)
        self.video_proj   = nn.Linear(input_dim, fusion_dim)

        # Scalar gate per modality (learned)
        self.text_gate    = nn.Linear(fusion_dim, 1)
        self.audio_gate   = nn.Linear(fusion_dim, 1)
        self.video_gate   = nn.Linear(fusion_dim, 1)

        self.activation   = nn.ReLU()
        self.dropout      = nn.Dropout(dropout)
        self.layer_norm   = nn.LayerNorm(fusion_dim)

        # Task heads
        self.emotion_head    = nn.Linear(fusion_dim, num_emotions)
        self.sentiment_head  = nn.Linear(fusion_dim, num_sentiments)

    def forward(self, text_feat=None, audio_feat=None, video_feat=None):
        fused = torch.zeros(1, 256)  # fallback (batch dim handled externally)
        gate_sum = 0.0

        modalities = []
        if text_feat is not None:
            t = self.activation(self.text_proj(text_feat))
            g = torch.sigmoid(self.text_gate(t))
            modalities.append((t, g))

        if audio_feat is not None:
            a = self.activation(self.audio_proj(audio_feat))
            g = torch.sigmoid(self.audio_gate(a))
            modalities.append((a, g))

        if video_feat is not None:
            v = self.activation(self.video_proj(video_feat))
            g = torch.sigmoid(self.video_gate(v))
            modalities.append((v, g))

        fused = sum(feat * gate for feat, gate in modalities)
        fused = self.layer_norm(self.dropout(fused))

        emotion_logits   = self.emotion_head(fused)
        sentiment_logits = self.sentiment_head(fused)
        return emotion_logits, sentiment_logits
```

### 4.2 Dimension Summary

| Layer | Input Shape | Output Shape |
|---|---|---|
| DistilRoBERTa (frozen) | `[B, 128]` token ids | `[B, 768]` |
| Wav2Vec2 (frozen) | `[B, T]` waveform | `[B, 768]` |
| ViT-base (frozen) | `[B×8, 3, 224, 224]` | `[B, 768]` |
| Text projection | `[B, 768]` | `[B, 256]` |
| Audio projection | `[B, 768]` | `[B, 256]` |
| Video projection | `[B, 768]` | `[B, 256]` |
| Text gate | `[B, 256]` | `[B, 1]` scalar |
| Audio gate | `[B, 256]` | `[B, 1]` scalar |
| Video gate | `[B, 256]` | `[B, 1]` scalar |
| Fused representation | — | `[B, 256]` |
| Emotion head | `[B, 256]` | `[B, 7]` logits |
| Sentiment head | `[B, 256]` | `[B, 3]` logits |

### 4.3 Total Trainable Parameters (Fusion Only)

| Component | Parameters |
|---|---|
| Text projection (`768→256`) | 196,864 |
| Audio projection (`768→256`) | 196,864 |
| Video projection (`768→256`) | 196,864 |
| Text gate (`256→1`) | 257 |
| Audio gate (`256→1`) | 257 |
| Video gate (`256→1`) | 257 |
| Layer norm | 512 |
| Emotion head (`256→7`) | 1,799 |
| Sentiment head (`256→3`) | 771 |
| **Total** | **~594K** |

> Only ~594K parameters need to be trained for the fusion model — well within free Colab GPU limits.

---

## 5. Feature Caching Strategy

Extracting features from 3 large pretrained encoders over 9,989 training utterances every epoch would be prohibitively slow on free Colab. Instead:

```
Training Phase 1: Extract features ONCE, save to Colab/Drive
Training Phase 2+: Load cached features, train only the fusion model

Cache files (saved to Google Drive):
  /content/drive/MyDrive/affectra/cache/
    text_train.pt      shape: [9989, 768]
    text_dev.pt        shape: [1109, 768]
    text_test.pt       shape: [2610, 768]
    audio_train.pt     shape: [9989, 768]
    audio_dev.pt       shape: [1109, 768]
    audio_test.pt      shape: [2610, 768]
    video_train.pt     shape: [9989, 768]
    video_dev.pt       shape: [1109, 768]
    video_test.pt      shape: [2610, 768]
```

### 5.1 Caching Logic

```python
import torch
import os

CACHE_DIR = "/content/drive/MyDrive/affectra/cache"

def save_cache(features: torch.Tensor, split: str, modality: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{modality}_{split}.pt")
    torch.save(features, path)
    print(f"Saved {modality} {split} cache: {features.shape} → {path}")

def load_cache(split: str, modality: str) -> torch.Tensor:
    path = os.path.join(CACHE_DIR, f"{modality}_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}. Run feature extraction first.")
    return torch.load(path)
```

---

## 6. Training Order (9 Phases)

### Phase 1 — Validate Dataset

```
Goal:   Confirm annotation CSVs loaded correctly
Action: Print split sizes, class distribution, sample rows
Output: Dataset validation report (printed in notebook)
```

### Phase 2 — 100-Sample Smoke Test

```
Goal:   Verify the full pipeline runs end-to-end without errors
Action: Extract features for 100 samples, run 1 epoch of fusion training
Output: Loss prints without crash (loss value doesn't matter yet)
```

### Phase 3 — Extract and Cache Text Features

```
Goal:   Cache DistilRoBERTa [CLS] embeddings for all splits
Action: Batch-process train/dev/test through frozen DistilRoBERTa
Output: text_train.pt, text_dev.pt, text_test.pt → saved to Drive
Time:   ~15–25 min on Colab T4
```

### Phase 4 — Extract and Cache Audio Features

```
Goal:   Cache Wav2Vec2-base mean-pooled embeddings for all splits
Action: Extract .mp4 audio → resample → encode → mean-pool → save
Output: audio_train.pt, audio_dev.pt, audio_test.pt → saved to Drive
Time:   ~40–60 min on Colab T4
Note:   Some utterances may have missing/corrupt audio — log and fill with zeros
```

### Phase 5 — Extract and Cache Video Features

```
Goal:   Cache ViT-base frame embeddings for all splits
Action: Decode .mp4 → sample 8 frames → encode → mean-pool CLS → save
Output: video_train.pt, video_dev.pt, video_test.pt → saved to Drive
Time:   ~60–90 min on Colab T4
Note:   Largest cache; mount Drive first to avoid losing data on runtime reset
```

### Phase 6 — Train Multimodal Fusion Model

```
Goal:   Train GatedMultimodalFusion on cached features
Action: Load cached .pt files → DataLoader → training loop → save best checkpoint
Config:
  optimizer:      AdamW, lr=2e-4, weight_decay=0.01
  loss:           weighted CrossEntropyLoss (handles class imbalance)
  epochs:         30 (with early stopping, patience=5)
  batch_size:     64
  scheduler:      ReduceLROnPlateau on dev weighted F1
  early stopping: monitor weighted F1 on dev split
Output: models/affectra_multimodal/model_state.pt (best checkpoint)
```

### Phase 7 — Evaluate on Dev Split

```
Goal:   Report full performance metrics on the official dev split
Action: Load best checkpoint → run inference on dev → compute metrics
Output: Weighted F1, per-class F1, confusion matrix, accuracy
Note:   This is the primary evaluation used during development
```

### Phase 8 — Final Test Evaluation (Run Once Only)

```
Goal:   Report final model performance on the held-out test split
Action: Load best checkpoint → run inference on test → compute metrics
Output: Final test metrics → saved to models/affectra_multimodal/metrics.json
RULE:   Do NOT run test evaluation multiple times. The test set is touched once.
```

### Phase 9 — Export Inference Artifacts

```
Goal:   Export everything needed to run the backend inference service
Action: Save all artifacts to models/affectra_multimodal/ → upload to Drive
Output:
  model_state.pt         - fusion model weights (not encoder weights)
  model_config.json      - architecture hyperparameters
  emotion_labels.json    - {0: "anger", 1: "disgust", ...}
  sentiment_labels.json  - {0: "positive", 1: "negative", 2: "neutral"}
  metrics.json           - final test metrics
  text_encoder/          - saved DistilRoBERTa tokenizer (for inference)
```

---

## 7. Training Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Better regularization than Adam |
| Learning rate | `2e-4` | For fusion layer only; encoders frozen |
| Weight decay | `0.01` | L2 regularization |
| Batch size | `64` | Fits Colab T4 with cached features |
| Max epochs | `30` | With early stopping |
| Early stopping patience | `5` epochs | Monitor dev weighted F1 |
| LR scheduler | `ReduceLROnPlateau` | Factor=0.5, patience=3 |
| Loss function | Weighted `CrossEntropyLoss` | Handles MELD class imbalance |
| Dropout | `0.3` | In fusion layer |
| Emotion loss weight | `0.6` | Alpha — primary task |
| Sentiment loss weight | `0.4` | Beta — auxiliary task |

---

## 8. Evaluation Metrics

| Metric | Description | Why Used |
|---|---|---|
| **Weighted F1** | F1 per class, weighted by support | Primary metric — handles class imbalance |
| **Macro F1** | Unweighted mean F1 across classes | Shows performance on minority classes |
| **Accuracy** | Overall correct predictions / total | Quick sanity check |
| **Per-class F1** | F1 for each emotion individually | Diagnose which emotions are hard |
| **Confusion matrix** | N×N grid of predicted vs actual | Visual debugging |

```python
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def evaluate(model, dataloader, device):
    model.eval()
    all_emotion_preds, all_emotion_labels = [], []
    all_sentiment_preds, all_sentiment_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            emotion_logits, sentiment_logits = model(...)
            all_emotion_preds.extend(emotion_logits.argmax(1).cpu().tolist())
            all_emotion_labels.extend(batch["emotion_label"].tolist())
            all_sentiment_preds.extend(sentiment_logits.argmax(1).cpu().tolist())
            all_sentiment_labels.extend(batch["sentiment_label"].tolist())

    emotion_wf1 = f1_score(all_emotion_labels, all_emotion_preds, average="weighted")
    emotion_mf1 = f1_score(all_emotion_labels, all_emotion_preds, average="macro")
    sentiment_wf1 = f1_score(all_sentiment_labels, all_sentiment_preds, average="weighted")

    return {
        "emotion_weighted_f1": emotion_wf1,
        "emotion_macro_f1": emotion_mf1,
        "sentiment_weighted_f1": sentiment_wf1,
        "emotion_accuracy": accuracy_score(all_emotion_labels, all_emotion_preds),
    }
```

---

## 9. Google Colab Workflow

### 9.1 Setup Checklist (Run Once per Session)

```python
# Step 1: Mount Google Drive (persist caches across sessions)
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone the repository
!git clone https://github.com/Pankaj429w63/Affectra-AI.git /content/Affectra-AI

# Step 3: Install dependencies
!pip install -q transformers torch torchaudio torchvision \
    opencv-python-headless librosa scikit-learn pandas tqdm

# Step 4: Verify GPU
import torch
print(torch.cuda.get_device_name(0))   # Expected: Tesla T4 or similar

# Step 5: Download MELD dataset
!wget -q -O /content/MELD.Raw.tar.gz \
    https://affective-meld.github.io/resources/MELD.Raw.tar.gz
!tar -xzf /content/MELD.Raw.tar.gz -C /content/meld_data/
```

### 9.2 Runtime Reset Survival

Colab free tier resets every ~12 hours. To survive this:

- **Mount Google Drive** before any long operation.
- **Save all feature caches to Drive** (`/content/drive/MyDrive/affectra/cache/`).
- **Save model checkpoint to Drive** after every epoch.
- **Check for existing cache files** at the start of each phase — skip extraction if already done.

### 9.3 Session Persistence Pattern

```python
def phase_guard(phase_name: str, cache_path: str):
    """Skip a phase if its output already exists in Drive."""
    if os.path.exists(cache_path):
        print(f"✅ Phase '{phase_name}' already complete. Skipping.")
        return True
    print(f"🔄 Running phase '{phase_name}'...")
    return False
```

---

## 10. Exported Model Artifacts

After Phase 9, the following artifacts are produced in `models/affectra_multimodal/`:

| File | Description | Used By |
|---|---|---|
| `model_state.pt` | Fusion model weights only (not encoder weights) | Backend inference |
| `model_config.json` | Fusion model hyperparameters (`input_dim`, `fusion_dim`, etc.) | Backend, reproducibility |
| `emotion_labels.json` | `{0: "anger", 1: "disgust", ...}` | Backend, frontend |
| `sentiment_labels.json` | `{0: "positive", 1: "negative", 2: "neutral"}` | Backend, frontend |
| `metrics.json` | Final test split metrics (F1, accuracy, per-class F1) | Documentation |
| `text_encoder/` | Saved DistilRoBERTa tokenizer (via `tokenizer.save_pretrained()`) | Backend text preprocessing |

### 10.1 Example `model_config.json`

```json
{
  "model_name": "GatedMultimodalFusion",
  "version": "1.0",
  "input_dim": 768,
  "fusion_dim": 256,
  "num_emotions": 7,
  "num_sentiments": 3,
  "dropout": 0.3,
  "text_encoder": "distilroberta-base",
  "audio_encoder": "facebook/wav2vec2-base",
  "video_encoder": "google/vit-base-patch16-224",
  "frames_per_clip": 8,
  "max_text_length": 128,
  "audio_sample_rate": 16000
}
```

### 10.2 Example `metrics.json`

```json
{
  "split": "test",
  "emotion_weighted_f1": 0.000,
  "emotion_macro_f1": 0.000,
  "emotion_accuracy": 0.000,
  "sentiment_weighted_f1": 0.000,
  "per_class_emotion_f1": {
    "anger": 0.000,
    "disgust": 0.000,
    "fear": 0.000,
    "joy": 0.000,
    "neutral": 0.000,
    "sadness": 0.000,
    "surprise": 0.000
  },
  "trained_on": "MELD official train split",
  "evaluated_on": "MELD official test split"
}
```

---

## 11. Training Notebook Location

```
training/notebooks/affectra_train.ipynb
```

The notebook is the single entry point for all 9 training phases. It is designed to run on Google Colab with a free GPU runtime.

---

*See [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md) for the full application architecture, deployment topology, and API contract.*
