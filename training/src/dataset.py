"""
Affectra AI — MELD Dataset Class
==================================
PyTorch Dataset that loads pre-extracted, cached features for training.

The dataset does NOT run the encoders (DistilRoBERTa, Wav2Vec2, ViT).
Encoders are run once during feature extraction (feature_extractors.py)
and their outputs are saved as .pt files in the feature cache directory.

This class simply loads those cached tensors and serves them to the
DataLoader in batches during training.

Supports:
  - text-only samples (audio/video features replaced with zeros)
  - text + audio
  - text + video
  - text + audio + video (full multimodal)

Missing modality caches → zero vectors with mask=0.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from training.src.config import (
    AUDIO_FEATURE_DIM,
    EMOTION_LABEL2ID,
    MELD_DEV_CSV,
    MELD_TEST_CSV,
    MELD_TRAIN_CSV,
    SENTIMENT_LABEL2ID,
    TEXT_FEATURE_DIM,
    VIDEO_FEATURE_DIM,
    normalise_emotion,
    normalise_sentiment,
)
from training.src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sample Metadata Loading
# ---------------------------------------------------------------------------

def load_meld_metadata(
    csv_path: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load MELD annotation CSV and return a cleaned DataFrame.

    Performs:
      - Loads the CSV
      - Normalises Emotion and Sentiment labels to lowercase
      - Creates integer ID columns (emotion_id, sentiment_id)
      - Creates a sample_id string: "dia{D}_utt{U}"
      - Optionally limits the number of rows (for smoke test)

    Args:
        csv_path:    Path to annotation CSV.
        max_samples: If set, only the first max_samples rows are returned.

    Returns:
        Cleaned pd.DataFrame with added columns:
          emotion_norm, emotion_id, sentiment_norm, sentiment_id, sample_id
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")

    if max_samples is not None:
        df = df.head(max_samples).copy()
        logger.info(f"Smoke test: limited to {max_samples} samples from {csv_path}")

    # Normalise labels
    df["emotion_norm"] = df["Emotion"].apply(
        lambda x: normalise_emotion(str(x))
    )
    df["sentiment_norm"] = df["Sentiment"].apply(
        lambda x: normalise_sentiment(str(x))
    )

    # Integer IDs
    df["emotion_id"] = df["emotion_norm"].map(EMOTION_LABEL2ID)
    df["sentiment_id"] = df["sentiment_norm"].map(SENTIMENT_LABEL2ID)

    # Sample identifier
    df["sample_id"] = df.apply(
        lambda r: f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}",
        axis=1,
    )

    logger.info(
        f"Loaded metadata: {len(df)} samples  "
        f"[emotions: {df['emotion_norm'].value_counts().to_dict()}]"
    )
    return df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MELDCachedDataset(Dataset):
    """
    PyTorch Dataset that serves cached multimodal features for MELD.

    Each sample returns a dictionary:
      {
        "text_feat":       FloatTensor [768]  — DistilRoBERTa CLS embedding
        "audio_feat":      FloatTensor [768]  — Wav2Vec2 mean-pool embedding
        "video_feat":      FloatTensor [768]  — ViT mean-pool frame embedding
        "text_mask":       FloatTensor [1]    — 1.0 if text available, else 0.0
        "audio_mask":      FloatTensor [1]    — 1.0 if audio available, else 0.0
        "video_mask":      FloatTensor [1]    — 1.0 if video available, else 0.0
        "emotion_label":   LongTensor  [1]    — 0–6 class index
        "sentiment_label": LongTensor  [1]    — 0–2 class index
        "sample_id":       str                — "dia{D}_utt{U}"
      }

    Args:
        metadata_df:    DataFrame from load_meld_metadata().
        text_features:  Optional FloatTensor [N, 768]. None → zero vectors.
        audio_features: Optional FloatTensor [N, 768]. None → zero vectors.
        video_features: Optional FloatTensor [N, 768]. None → zero vectors.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        text_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        n = len(self.metadata)

        # Validate feature tensor lengths
        for name, feats in [
            ("text", text_features),
            ("audio", audio_features),
            ("video", video_features),
        ]:
            if feats is not None and len(feats) != n:
                raise ValueError(
                    f"{name}_features has {len(feats)} rows, "
                    f"but metadata has {n} rows. They must match."
                )

        # Store features or mark as unavailable
        self.text_features = text_features    # [N, 768] or None
        self.audio_features = audio_features  # [N, 768] or None
        self.video_features = video_features  # [N, 768] or None

        # Pre-build zero tensors for missing modalities
        self._zero_text = torch.zeros(TEXT_FEATURE_DIM, dtype=torch.float32)
        self._zero_audio = torch.zeros(AUDIO_FEATURE_DIM, dtype=torch.float32)
        self._zero_video = torch.zeros(VIDEO_FEATURE_DIM, dtype=torch.float32)

        logger.info(
            f"MELDCachedDataset: {n} samples | "
            f"text={'✅' if text_features is not None else '❌'} | "
            f"audio={'✅' if audio_features is not None else '❌'} | "
            f"video={'✅' if video_features is not None else '❌'}"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]

        # Text feature + mask
        if self.text_features is not None:
            text_feat = self.text_features[idx].float()
            text_mask = torch.tensor(1.0)
        else:
            text_feat = self._zero_text.clone()
            text_mask = torch.tensor(0.0)

        # Audio feature + mask
        if self.audio_features is not None:
            audio_feat = self.audio_features[idx].float()
            audio_mask = torch.tensor(1.0)
        else:
            audio_feat = self._zero_audio.clone()
            audio_mask = torch.tensor(0.0)

        # Video feature + mask
        if self.video_features is not None:
            video_feat = self.video_features[idx].float()
            video_mask = torch.tensor(1.0)
        else:
            video_feat = self._zero_video.clone()
            video_mask = torch.tensor(0.0)

        return {
            "text_feat":       text_feat,
            "audio_feat":      audio_feat,
            "video_feat":      video_feat,
            "text_mask":       text_mask,
            "audio_mask":      audio_mask,
            "video_mask":      video_mask,
            "emotion_label":   torch.tensor(int(row["emotion_id"]), dtype=torch.long),
            "sentiment_label": torch.tensor(int(row["sentiment_id"]), dtype=torch.long),
            "sample_id":       str(row["sample_id"]),
        }


# ---------------------------------------------------------------------------
# DataLoader Builder
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: MELDCachedDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader with the appropriate settings for Colab.

    Args:
        dataset:     MELDCachedDataset instance.
        batch_size:  Number of samples per batch.
        shuffle:     Whether to shuffle (True for train, False for dev/test).
        num_workers: Parallel data loading workers (2 is safe for Colab).
        pin_memory:  Pin memory for faster GPU transfer (True if GPU available).

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Convenience Builder
# ---------------------------------------------------------------------------

def build_split_dataset(
    meld_root: str,
    split: str,
    text_feats: Optional[torch.Tensor],
    audio_feats: Optional[torch.Tensor],
    video_feats: Optional[torch.Tensor],
    max_samples: Optional[int] = None,
) -> MELDCachedDataset:
    """
    Convenience function: load metadata for a split and build a dataset.

    Args:
        meld_root:    Directory containing annotation CSVs.
        split:        One of 'train', 'dev', 'test'.
        text_feats:   Cached text features tensor or None.
        audio_feats:  Cached audio features tensor or None.
        video_feats:  Cached video features tensor or None.
        max_samples:  Limit rows (for smoke test).

    Returns:
        MELDCachedDataset ready for DataLoader.
    """
    csv_map = {
        "train": MELD_TRAIN_CSV,
        "dev": MELD_DEV_CSV,
        "test": MELD_TEST_CSV,
    }
    if split not in csv_map:
        raise ValueError(f"Unknown split: '{split}'. Must be 'train', 'dev', or 'test'.")

    csv_path = os.path.join(meld_root, csv_map[split])
    df = load_meld_metadata(csv_path, max_samples=max_samples)

    return MELDCachedDataset(
        metadata_df=df,
        text_features=text_feats,
        audio_features=audio_feats,
        video_features=video_feats,
    )
