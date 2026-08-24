"""
Affectra AI — Model Export
============================
Exports the trained GatedMultimodalFusion model and all inference
artifacts to the canonical output directory:

  models/affectra_multimodal/
    ├── model_state.pt          ← fusion model weights only
    ├── model_config.json       ← architecture + training config
    ├── emotion_labels.json     ← {0: "anger", 1: "disgust", ...}
    ├── sentiment_labels.json   ← {0: "positive", 1: "negative", 2: "neutral"}
    ├── metrics.json            ← final test evaluation results
    └── text_encoder/           ← saved DistilRoBERTa tokenizer

IMPORTANT safety rules:
  - Only model.state_dict() is saved — never the full model object.
  - No pickle of Python objects.
  - The tokenizer is saved with tokenizer.save_pretrained() (HuggingFace standard).
  - Encoder weights (DistilRoBERTa, Wav2Vec2, ViT) are NOT saved —
    they are loaded from HuggingFace at inference time.
"""

import os
import shutil
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from training.src.config import (
    COLAB_CHECKPOINT_DIR,
    COLAB_DRIVE_ROOT,
    EMOTION_ID2LABEL,
    EMOTION_LABELS,
    METRICS_FILENAME,
    MODEL_ARTIFACT_DIR,
    MODEL_CONFIG_FILENAME,
    MODEL_STATE_FILENAME,
    EMOTION_LABELS_FILENAME,
    SENTIMENT_LABELS_FILENAME,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABELS,
    TEXT_ENCODER_NAME,
    TEXT_ENCODER_SUBDIR,
    get_model_config,
)
from training.src.utils import get_logger, save_json

logger = get_logger(__name__)

# Drive-side export directory (for backup before copying to repo)
DRIVE_EXPORT_DIR = f"{COLAB_DRIVE_ROOT}/training_outputs/affectra_multimodal"


# ---------------------------------------------------------------------------
# Export model weights
# ---------------------------------------------------------------------------

def export_model_weights(
    model: nn.Module,
    output_dir: str,
) -> str:
    """
    Save the fusion model's state dict as model_state.pt.

    ONLY the fusion model's state dict is saved.
    Encoder weights (DistilRoBERTa, Wav2Vec2, ViT) are loaded from
    HuggingFace Hub at inference time — they must NOT be included here.

    Args:
        model:      Trained GatedMultimodalFusion instance.
        output_dir: Directory to save into.

    Returns:
        str: Full path to the saved model_state.pt file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MODEL_STATE_FILENAME)

    # Save state dict only — never use torch.save(model, path)
    torch.save(model.state_dict(), path)

    size_mb = os.path.getsize(path) / (1024 ** 2)
    logger.info(f"✅ Model weights saved: {path}  ({size_mb:.1f} MB)")
    return path


# ---------------------------------------------------------------------------
# Export model config
# ---------------------------------------------------------------------------

def export_model_config(output_dir: str) -> str:
    """
    Save model_config.json with architecture and training hyperparameters.

    Args:
        output_dir: Directory to save into.

    Returns:
        str: Full path to the saved model_config.json file.
    """
    path = os.path.join(output_dir, MODEL_CONFIG_FILENAME)
    config = get_model_config()
    save_json(config, path)
    logger.info(f"✅ Model config saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Export label mappings
# ---------------------------------------------------------------------------

def export_label_mappings(output_dir: str) -> tuple:
    """
    Save emotion_labels.json and sentiment_labels.json.

    The JSON format is: {"0": "anger", "1": "disgust", ...}
    (string keys for JSON compatibility)

    Args:
        output_dir: Directory to save into.

    Returns:
        Tuple[str, str]: paths to emotion and sentiment label files.
    """
    emo_path  = os.path.join(output_dir, EMOTION_LABELS_FILENAME)
    sent_path = os.path.join(output_dir, SENTIMENT_LABELS_FILENAME)

    # Use string keys for JSON standard compliance
    emo_map  = {str(k): v for k, v in EMOTION_ID2LABEL.items()}
    sent_map = {str(k): v for k, v in SENTIMENT_ID2LABEL.items()}

    save_json(emo_map, emo_path)
    save_json(sent_map, sent_path)

    logger.info(f"✅ Emotion labels saved:   {emo_path}")
    logger.info(f"✅ Sentiment labels saved: {sent_path}")
    return emo_path, sent_path


# ---------------------------------------------------------------------------
# Export tokenizer
# ---------------------------------------------------------------------------

def export_text_tokenizer(output_dir: str) -> str:
    """
    Save the DistilRoBERTa tokenizer for use in the FastAPI backend.

    Uses HuggingFace's standard save_pretrained() method.
    This saves the tokenizer vocab and config files (not model weights).

    Args:
        output_dir: Parent directory (tokenizer will be saved in text_encoder/ subdir).

    Returns:
        str: Path to the saved tokenizer directory.
    """
    tokenizer_dir = os.path.join(output_dir, TEXT_ENCODER_SUBDIR)
    os.makedirs(tokenizer_dir, exist_ok=True)

    logger.info(f"Loading tokenizer: {TEXT_ENCODER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
    tokenizer.save_pretrained(tokenizer_dir)

    logger.info(f"✅ Tokenizer saved: {tokenizer_dir}")
    return tokenizer_dir


# ---------------------------------------------------------------------------
# Export metrics
# ---------------------------------------------------------------------------

def export_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    trained_on: str = "MELD official train split",
    evaluated_on: str = "MELD official test split",
) -> str:
    """
    Save evaluation metrics to metrics.json.

    Args:
        metrics:       Metrics dict from evaluate.evaluate().
        output_dir:    Directory to save into.
        trained_on:    Description of training set used.
        evaluated_on:  Description of evaluation set used.

    Returns:
        str: Full path to saved metrics.json.
    """
    metrics_out = dict(metrics)  # Shallow copy
    metrics_out["trained_on"]   = trained_on
    metrics_out["evaluated_on"] = evaluated_on

    path = os.path.join(output_dir, METRICS_FILENAME)
    save_json(metrics_out, path)
    logger.info(f"✅ Metrics saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Full export pipeline
# ---------------------------------------------------------------------------

def export_all(
    model: nn.Module,
    test_metrics: Dict[str, Any],
    output_dir: str = MODEL_ARTIFACT_DIR,
    also_save_to_drive: bool = True,
) -> Dict[str, str]:
    """
    Export all inference artifacts to the canonical output directory.

    This is the single function to call after final test evaluation.
    It exports:
      - model_state.pt
      - model_config.json
      - emotion_labels.json
      - sentiment_labels.json
      - metrics.json
      - text_encoder/

    Args:
        model:             Trained GatedMultimodalFusion model.
        test_metrics:      Metrics from final test evaluation.
        output_dir:        Local output directory (default: models/affectra_multimodal/).
        also_save_to_drive: Also copy all artifacts to Google Drive for safety.

    Returns:
        Dict mapping artifact name to its saved path.
    """
    logger.info("=" * 55)
    logger.info(f"EXPORTING AFFECTRA AI INFERENCE ARTIFACTS")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # 1. Model weights
    paths["model_state"]    = export_model_weights(model, output_dir)

    # 2. Model config
    paths["model_config"]   = export_model_config(output_dir)

    # 3. Label mappings
    emo_path, sent_path     = export_label_mappings(output_dir)
    paths["emotion_labels"]   = emo_path
    paths["sentiment_labels"] = sent_path

    # 4. Tokenizer
    paths["text_encoder"]   = export_text_tokenizer(output_dir)

    # 5. Metrics
    paths["metrics"]        = export_metrics(test_metrics, output_dir)

    # 6. Optionally mirror to Google Drive
    if also_save_to_drive:
        try:
            logger.info(f"Copying artifacts to Google Drive: {DRIVE_EXPORT_DIR}")
            if os.path.exists(DRIVE_EXPORT_DIR):
                shutil.rmtree(DRIVE_EXPORT_DIR)
            shutil.copytree(output_dir, DRIVE_EXPORT_DIR)
            logger.info(f"✅ Artifacts mirrored to Drive: {DRIVE_EXPORT_DIR}")
        except Exception as e:
            logger.warning(
                f"Could not copy to Drive: {e}. "
                "Artifacts are still saved locally."
            )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n=== EXPORT COMPLETE ===")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    logger.info(
        "\nNext steps:\n"
        "  1. Download models/affectra_multimodal/ from Colab or Drive.\n"
        "  2. Place it at: Affectra-AI/models/affectra_multimodal/\n"
        "  3. The FastAPI backend will load model_state.pt from MODEL_DIR.\n"
    )

    return paths
