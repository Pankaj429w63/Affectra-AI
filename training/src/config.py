"""
Affectra AI — Central Training Configuration
============================================
Single source of truth for all paths, labels, hyperparameters,
and encoder names used across the training pipeline.

This module is designed to run both locally (for inspection) and
inside Google Colab. Colab-specific paths are clearly marked.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# MELD Official Labels
# Do NOT change these — they are the exact labels in the CSV files.
# ---------------------------------------------------------------------------

EMOTION_LABELS: List[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

SENTIMENT_LABELS: List[str] = [
    "positive",
    "negative",
    "neutral",
]

# Integer → label mapping (used for JSON export and metric reporting)
EMOTION_ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(EMOTION_LABELS)}
EMOTION_LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}

SENTIMENT_ID2LABEL: Dict[int, str] = {i: lbl for i, lbl in enumerate(SENTIMENT_LABELS)}
SENTIMENT_LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(SENTIMENT_LABELS)}

NUM_EMOTION_CLASSES: int = len(EMOTION_LABELS)
NUM_SENTIMENT_CLASSES: int = len(SENTIMENT_LABELS)

# ---------------------------------------------------------------------------
# Label normalisation helpers
# MELD CSVs use mixed-case labels (e.g. "Neutral", "Joy").
# Always normalise to lowercase before creating numeric IDs.
# ---------------------------------------------------------------------------

# Known MELD CSV label variants → canonical lowercase
_EMOTION_ALIASES: Dict[str, str] = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "joy": "joy",
    "happy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
}

_SENTIMENT_ALIASES: Dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


def normalise_emotion(raw: str) -> str:
    """
    Convert a raw MELD CSV emotion label to canonical lowercase form.
    Raises ValueError if the label is completely unrecognised.
    """
    key = raw.strip().lower()
    if key in _EMOTION_ALIASES:
        return _EMOTION_ALIASES[key]
    raise ValueError(
        f"Unrecognised emotion label: '{raw}'. "
        f"Expected one of: {list(_EMOTION_ALIASES.keys())}"
    )


def normalise_sentiment(raw: str) -> str:
    """
    Convert a raw MELD CSV sentiment label to canonical lowercase form.
    Raises ValueError if the label is completely unrecognised.
    """
    key = raw.strip().lower()
    if key in _SENTIMENT_ALIASES:
        return _SENTIMENT_ALIASES[key]
    raise ValueError(
        f"Unrecognised sentiment label: '{raw}'. "
        f"Expected one of: {list(_SENTIMENT_ALIASES.keys())}"
    )


# ---------------------------------------------------------------------------
# Pretrained Encoder Identifiers
# ---------------------------------------------------------------------------

TEXT_ENCODER_NAME: str = "distilroberta-base"
AUDIO_ENCODER_NAME: str = "facebook/wav2vec2-base"
VIDEO_ENCODER_NAME: str = "google/vit-base-patch16-224"

# ---------------------------------------------------------------------------
# Encoder Output Dimensions
# ---------------------------------------------------------------------------

TEXT_FEATURE_DIM: int = 768    # DistilRoBERTa hidden size
AUDIO_FEATURE_DIM: int = 768   # Wav2Vec2-base hidden size
VIDEO_FEATURE_DIM: int = 768   # ViT-base hidden size
FUSION_DIM: int = 256          # Projected fusion dimension

# ---------------------------------------------------------------------------
# Encoder Preprocessing Parameters
# ---------------------------------------------------------------------------

TEXT_MAX_LENGTH: int = 128          # Max tokens — MELD utterances are short
AUDIO_SAMPLE_RATE: int = 16_000     # Wav2Vec2 requires 16 kHz mono
VIDEO_FRAMES_PER_CLIP: int = 8      # Uniformly sampled frames per video

# ---------------------------------------------------------------------------
# Fusion Model Hyperparameters
# ---------------------------------------------------------------------------

DROPOUT: float = 0.3

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE: float = 2e-4
WEIGHT_DECAY: float = 0.01
BATCH_SIZE: int = 64            # Reduce to 32 if you see CUDA OOM errors
MAX_EPOCHS: int = 30
EARLY_STOPPING_PATIENCE: int = 5
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 3

# Multi-task loss weights (must be positive; do not need to sum to 1)
ALPHA_EMOTION: float = 0.6      # Weight for emotion classification loss
BETA_SENTIMENT: float = 0.4     # Weight for sentiment classification loss

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Smoke-Test Sample Counts
# ---------------------------------------------------------------------------

SMOKE_TRAIN_N: int = 100
SMOKE_DEV_N: int = 30
SMOKE_TEST_N: int = 30

# ---------------------------------------------------------------------------
# Colab Paths
# All paths below are absolute paths inside Google Colab.
# They are not valid on your local Windows machine.
# ---------------------------------------------------------------------------

COLAB_REPO_DIR: str = "/content/Affectra-AI"
COLAB_DATA_DIR: str = "/content/meld_data"
COLAB_DRIVE_ROOT: str = "/content/drive/MyDrive/AffectraAI"
COLAB_CACHE_DIR: str = f"{COLAB_DRIVE_ROOT}/feature_cache"
COLAB_CHECKPOINT_DIR: str = f"{COLAB_DRIVE_ROOT}/checkpoints"
COLAB_LOG_DIR: str = f"{COLAB_DRIVE_ROOT}/logs"
COLAB_OUTPUT_DIR: str = f"{COLAB_DRIVE_ROOT}/training_outputs"

# Repo-relative path for exported model artifacts (synced from Colab → Drive → here)
MODEL_ARTIFACT_DIR: str = "./models/affectra_multimodal"
MODEL_STATE_FILENAME: str = "model_state.pt"
MODEL_CONFIG_FILENAME: str = "model_config.json"
EMOTION_LABELS_FILENAME: str = "emotion_labels.json"
SENTIMENT_LABELS_FILENAME: str = "sentiment_labels.json"
METRICS_FILENAME: str = "metrics.json"
TEXT_ENCODER_SUBDIR: str = "text_encoder"

# ---------------------------------------------------------------------------
# MELD Archive Download
# ---------------------------------------------------------------------------

# Download mirrors, tried IN ORDER until one succeeds (see download.py).
# Both serve the byte-identical archive — verified live: Content-Length
# 10,878,146,150 bytes (~10.13 GiB) and gzip magic 1f8b on each.
#
# HuggingFace is listed FIRST because it is a reliable CDN reachable from Colab.
# The umich host is kept as a backup: it intermittently refuses connections from
# Colab (curl returns HTTP 000 — a connection failure, not a real HTTP status),
# which is exactly why a single-URL download was fragile.
MELD_RAW_URLS: List[str] = [
    "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
    "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
]

# Backward-compatible alias — the primary (first) mirror.
MELD_RAW_URL: str = MELD_RAW_URLS[0]

# Official annotation repository (for CSV files)
MELD_ANNOTATION_REPO: str = "https://github.com/declare-lab/MELD.git"

# Expected annotation CSV filenames (relative to MELD root after extraction)
MELD_TRAIN_CSV: str = "train_sent_emo.csv"
MELD_DEV_CSV: str = "dev_sent_emo.csv"
MELD_TEST_CSV: str = "test_sent_emo.csv"

# Video filename pattern — ALWAYS use this pattern
# dia{Dialogue_ID}_utt{Utterance_ID}.mp4
MELD_VIDEO_FILENAME_PATTERN: str = "dia{dialogue_id}_utt{utterance_id}.mp4"

# ---------------------------------------------------------------------------
# Model Configuration Dictionary (for JSON serialisation)
# ---------------------------------------------------------------------------

def get_model_config() -> dict:
    """
    Returns a serialisable dictionary describing the full model configuration.
    Saved as model_config.json alongside model_state.pt.
    """
    return {
        "model_name": "GatedMultimodalFusion",
        "version": "1.0",
        "input_dim": TEXT_FEATURE_DIM,
        "fusion_dim": FUSION_DIM,
        "num_emotions": NUM_EMOTION_CLASSES,
        "num_sentiments": NUM_SENTIMENT_CLASSES,
        "dropout": DROPOUT,
        "text_encoder": TEXT_ENCODER_NAME,
        "audio_encoder": AUDIO_ENCODER_NAME,
        "video_encoder": VIDEO_ENCODER_NAME,
        "frames_per_clip": VIDEO_FRAMES_PER_CLIP,
        "max_text_length": TEXT_MAX_LENGTH,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "training": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "alpha_emotion": ALPHA_EMOTION,
            "beta_sentiment": BETA_SENTIMENT,
            "random_seed": RANDOM_SEED,
        },
    }


if __name__ == "__main__":
    import json
    print("=== Affectra AI Config ===")
    print(f"Emotion labels ({NUM_EMOTION_CLASSES}): {EMOTION_LABELS}")
    print(f"Sentiment labels ({NUM_SENTIMENT_CLASSES}): {SENTIMENT_LABELS}")
    print(f"Text encoder:  {TEXT_ENCODER_NAME}")
    print(f"Audio encoder: {AUDIO_ENCODER_NAME}")
    print(f"Video encoder: {VIDEO_ENCODER_NAME}")
    print("\nModel config:")
    print(json.dumps(get_model_config(), indent=2))
