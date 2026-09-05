"""
Affectra AI — Feature Cache
=============================
Handles saving and loading of extracted feature tensors to/from
Google Drive (or any directory on disk).

Feature caches are stored as PyTorch .pt files:
  {modality}_{split}.pt  →  FloatTensor [N, 768]

Alongside each feature file, a metadata sidecar is saved:
  {modality}_{split}_meta.json  →  records encoder name, shape, date

The metadata is verified before loading to ensure the cached features
were produced with the same encoder configuration. This prevents
accidentally mixing features from different model versions.

Cache structure on Google Drive:
  /content/drive/MyDrive/AffectraAI/feature_cache/
    text_train.pt
    text_train_meta.json
    text_dev.pt
    text_dev_meta.json
    text_test.pt
    text_test_meta.json
    audio_train.pt   ...
    video_train.pt   ...
"""

import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import torch

from training.src.config import (
    AUDIO_ENCODER_NAME,
    AUDIO_FEATURE_DIM,
    COLAB_CACHE_DIR,
    TEXT_ENCODER_NAME,
    TEXT_FEATURE_DIM,
    VIDEO_ENCODER_NAME,
    VIDEO_FEATURE_DIM,
)
from training.src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected encoder → feature dim mapping (for metadata verification)
# ---------------------------------------------------------------------------

_MODALITY_CONFIG = {
    "text": {
        "encoder": TEXT_ENCODER_NAME,
        "feature_dim": TEXT_FEATURE_DIM,
    },
    "audio": {
        "encoder": AUDIO_ENCODER_NAME,
        "feature_dim": AUDIO_FEATURE_DIM,
    },
    "video": {
        "encoder": VIDEO_ENCODER_NAME,
        "feature_dim": VIDEO_FEATURE_DIM,
    },
}

VALID_MODALITIES = set(_MODALITY_CONFIG.keys())
VALID_SPLITS = {"train", "dev", "test"}


# ---------------------------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------------------------

def _feature_path(modality: str, split: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{modality}_{split}.pt")


def _meta_path(modality: str, split: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{modality}_{split}_meta.json")


def _validate_args(modality: str, split: str) -> None:
    if modality not in VALID_MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'. Must be one of: {VALID_MODALITIES}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Must be one of: {VALID_SPLITS}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_features(
    features: torch.Tensor,
    modality: str,
    split: str,
    cache_dir: str = COLAB_CACHE_DIR,
    sample_ids: Optional[list] = None,
) -> str:
    """
    Save a feature tensor and its metadata sidecar to disk.

    Args:
        features:   FloatTensor [N, feature_dim].
        modality:   'text', 'audio', or 'video'.
        split:      'train', 'dev', or 'test'.
        cache_dir:  Directory to save into (default: Colab Drive path).
        sample_ids: Optional list of sample ID strings for debugging.

    Returns:
        str: Path of the saved .pt file.

    Raises:
        ValueError: If modality/split is invalid or features have wrong shape.
    """
    _validate_args(modality, split)

    expected_dim = _MODALITY_CONFIG[modality]["feature_dim"]
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Expected features shape [N, {expected_dim}], got {features.shape}"
        )

    os.makedirs(cache_dir, exist_ok=True)

    pt_path = _feature_path(modality, split, cache_dir)
    meta_path = _meta_path(modality, split, cache_dir)

    # Save tensor
    torch.save(features.cpu().float(), pt_path)

    # Save metadata sidecar
    meta = {
        "modality": modality,
        "split": split,
        "encoder": _MODALITY_CONFIG[modality]["encoder"],
        "feature_dim": expected_dim,
        "num_samples": features.shape[0],
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if sample_ids is not None:
        meta["sample_ids_head"] = sample_ids[:5]  # Store first 5 for inspection

    save_json(meta, meta_path)

    size_mb = os.path.getsize(pt_path) / (1024 ** 2)
    logger.info(
        f"💾 Cache saved: {pt_path}  "
        f"shape={list(features.shape)}  size={size_mb:.1f} MB"
    )
    return pt_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_features(
    modality: str,
    split: str,
    cache_dir: str = COLAB_CACHE_DIR,
    verify_metadata: bool = True,
) -> Optional[torch.Tensor]:
    """
    Load cached feature tensor from disk, optionally verifying metadata.

    Args:
        modality:        'text', 'audio', or 'video'.
        split:           'train', 'dev', or 'test'.
        cache_dir:       Directory to load from.
        verify_metadata: If True, check encoder name matches current config.

    Returns:
        FloatTensor [N, feature_dim], or None if cache does not exist.

    Raises:
        ValueError: If metadata verification fails (encoder mismatch).
    """
    _validate_args(modality, split)

    pt_path = _feature_path(modality, split, cache_dir)
    meta_path = _meta_path(modality, split, cache_dir)

    if not os.path.exists(pt_path):
        logger.info(f"Cache not found: {pt_path}")
        return None

    # Load and verify metadata
    if verify_metadata and os.path.exists(meta_path):
        meta = load_json(meta_path)
        expected_encoder = _MODALITY_CONFIG[modality]["encoder"]
        cached_encoder = meta.get("encoder", "")

        if cached_encoder != expected_encoder:
            raise ValueError(
                f"Cache metadata mismatch for {modality}/{split}!\n"
                f"  Cached encoder:  '{cached_encoder}'\n"
                f"  Current config:  '{expected_encoder}'\n"
                "Delete the cache and re-extract features with the current encoder."
            )

        expected_dim = _MODALITY_CONFIG[modality]["feature_dim"]
        cached_dim = meta.get("feature_dim", 0)
        if cached_dim != expected_dim:
            raise ValueError(
                f"Cache feature_dim mismatch for {modality}/{split}: "
                f"cached={cached_dim}, expected={expected_dim}"
            )

        logger.info(
            f"📂 Loading cache: {pt_path}  "
            f"shape={meta['shape']}  encoder='{cached_encoder}'"
        )
    else:
        logger.info(f"📂 Loading cache (no metadata): {pt_path}")

    features = torch.load(pt_path, map_location="cpu")

    size_mb = os.path.getsize(pt_path) / (1024 ** 2)
    logger.info(
        f"✅ Loaded {modality}/{split}: shape={list(features.shape)}  "
        f"size={size_mb:.1f} MB"
    )
    return features.float()


# ---------------------------------------------------------------------------
# Cache Status
# ---------------------------------------------------------------------------

def cache_status(cache_dir: str = COLAB_CACHE_DIR) -> Dict[str, Dict[str, bool]]:
    """
    Print and return the status of all expected cache files.

    Returns:
        Nested dict: {modality: {split: exists}}
    """
    status = {}
    logger.info(f"=== Feature Cache Status ({cache_dir}) ===")

    for modality in ["text", "audio", "video"]:
        status[modality] = {}
        for split in ["train", "dev", "test"]:
            pt_path = _feature_path(modality, split, cache_dir)
            meta_path = _meta_path(modality, split, cache_dir)
            exists = os.path.exists(pt_path)
            has_meta = os.path.exists(meta_path)

            status[modality][split] = exists
            icon = "✅" if exists else "❌"
            meta_icon = "(+meta)" if has_meta else "(no meta)"

            size_str = ""
            if exists:
                size_mb = os.path.getsize(pt_path) / (1024 ** 2)
                size_str = f"  {size_mb:.0f} MB"

            logger.info(f"  {icon} {modality}/{split}{size_str}  {meta_icon}")

    return status


# ---------------------------------------------------------------------------
# Load All Splits for One Modality
# ---------------------------------------------------------------------------

def load_all_splits(
    modality: str,
    cache_dir: str = COLAB_CACHE_DIR,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load train/dev/test caches for a single modality in one call.

    Returns:
        Tuple of (train_feats, dev_feats, test_feats).
        Each is a FloatTensor [N, 768] or None if not cached.
    """
    train = load_features(modality, "train", cache_dir)
    dev = load_features(modality, "dev", cache_dir)
    test = load_features(modality, "test", cache_dir)
    return train, dev, test
