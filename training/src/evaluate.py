"""
Affectra AI — Evaluation
==========================
Computes full evaluation metrics for both emotion and sentiment tasks.

Metrics computed:
  - Accuracy
  - Weighted Precision, Recall, F1
  - Macro Precision, Recall, F1
  - Per-class F1 for each emotion and sentiment label
  - Confusion matrix (as nested list for JSON serialisation)

Design:
  - Accepts a DataLoader and a trained GatedMultimodalFusion model.
  - Runs in eval() mode with torch.no_grad().
  - Returns a structured dict suitable for saving as metrics.json.
  - Never modifies the model's weights.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.src.config import (
    EMOTION_ID2LABEL,
    EMOTION_LABELS,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABELS,
)
from training.src.utils import get_logger, save_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Run inference on a DataLoader and collect predictions + labels.

    Args:
        model:      Trained GatedMultimodalFusion (in eval() mode).
        dataloader: DataLoader over a MELDCachedDataset split.
        device:     torch.device.

    Returns:
        Tuple of four lists:
          (emotion_preds, emotion_labels, sentiment_preds, sentiment_labels)
        All are integer class indices.
    """
    model.eval()

    all_emo_preds: List[int] = []
    all_emo_labels: List[int] = []
    all_sent_preds: List[int] = []
    all_sent_labels: List[int] = []

    for batch in dataloader:
        # Move tensors to device
        text_feat      = batch["text_feat"].to(device)
        audio_feat     = batch["audio_feat"].to(device)
        video_feat     = batch["video_feat"].to(device)
        text_mask      = batch["text_mask"].to(device)
        audio_mask     = batch["audio_mask"].to(device)
        video_mask     = batch["video_mask"].to(device)
        emotion_label  = batch["emotion_label"]    # keep on CPU for sklearn
        sentiment_label = batch["sentiment_label"]

        # Forward pass
        emo_logits, sent_logits = model(
            text_feat, audio_feat, video_feat,
            text_mask, audio_mask, video_mask,
        )

        # Predictions (argmax)
        emo_preds  = emo_logits.argmax(dim=1).cpu().tolist()
        sent_preds = sent_logits.argmax(dim=1).cpu().tolist()

        all_emo_preds.extend(emo_preds)
        all_emo_labels.extend(emotion_label.tolist())
        all_sent_preds.extend(sent_preds)
        all_sent_labels.extend(sentiment_label.tolist())

    return all_emo_preds, all_emo_labels, all_sent_preds, all_sent_labels


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    emo_preds: List[int],
    emo_labels: List[int],
    sent_preds: List[int],
    sent_labels: List[int],
    split: str,
) -> Dict[str, Any]:
    """
    Compute all classification metrics for emotion and sentiment tasks.

    Args:
        emo_preds:    Predicted emotion class indices.
        emo_labels:   True emotion class indices.
        sent_preds:   Predicted sentiment class indices.
        sent_labels:  True sentiment class indices.
        split:        'train', 'dev', or 'test' (for reporting).

    Returns:
        Structured metrics dictionary.
    """
    n_emo_classes  = len(EMOTION_LABELS)
    n_sent_classes = len(SENTIMENT_LABELS)

    # ── Emotion metrics ───────────────────────────────────────────────────
    emo_acc     = accuracy_score(emo_labels, emo_preds)
    emo_wf1     = f1_score(emo_labels, emo_preds, average="weighted", zero_division=0)
    emo_mf1     = f1_score(emo_labels, emo_preds, average="macro", zero_division=0)
    emo_wp      = precision_score(emo_labels, emo_preds, average="weighted", zero_division=0)
    emo_wr      = recall_score(emo_labels, emo_preds, average="weighted", zero_division=0)

    emo_per_class_f1 = f1_score(
        emo_labels, emo_preds,
        average=None,
        labels=list(range(n_emo_classes)),
        zero_division=0,
    )
    emo_cm = confusion_matrix(
        emo_labels, emo_preds,
        labels=list(range(n_emo_classes)),
    ).tolist()

    # ── Sentiment metrics ─────────────────────────────────────────────────
    sent_acc = accuracy_score(sent_labels, sent_preds)
    sent_wf1 = f1_score(sent_labels, sent_preds, average="weighted", zero_division=0)
    sent_mf1 = f1_score(sent_labels, sent_preds, average="macro", zero_division=0)
    sent_wp  = precision_score(sent_labels, sent_preds, average="weighted", zero_division=0)
    sent_wr  = recall_score(sent_labels, sent_preds, average="weighted", zero_division=0)

    sent_per_class_f1 = f1_score(
        sent_labels, sent_preds,
        average=None,
        labels=list(range(n_sent_classes)),
        zero_division=0,
    )
    sent_cm = confusion_matrix(
        sent_labels, sent_preds,
        labels=list(range(n_sent_classes)),
    ).tolist()

    # ── Build report dict ─────────────────────────────────────────────────
    metrics = {
        "split": split,
        "num_samples": len(emo_labels),

        "emotion": {
            "accuracy":        round(float(emo_acc), 4),
            "weighted_f1":     round(float(emo_wf1), 4),
            "macro_f1":        round(float(emo_mf1), 4),
            "weighted_precision": round(float(emo_wp), 4),
            "weighted_recall":    round(float(emo_wr), 4),
            "per_class_f1": {
                EMOTION_ID2LABEL[i]: round(float(emo_per_class_f1[i]), 4)
                for i in range(n_emo_classes)
            },
            "confusion_matrix": emo_cm,
            "class_names": EMOTION_LABELS,
        },

        "sentiment": {
            "accuracy":        round(float(sent_acc), 4),
            "weighted_f1":     round(float(sent_wf1), 4),
            "macro_f1":        round(float(sent_mf1), 4),
            "weighted_precision": round(float(sent_wp), 4),
            "weighted_recall":    round(float(sent_wr), 4),
            "per_class_f1": {
                SENTIMENT_ID2LABEL[i]: round(float(sent_per_class_f1[i]), 4)
                for i in range(n_sent_classes)
            },
            "confusion_matrix": sent_cm,
            "class_names": SENTIMENT_LABELS,
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Log metrics to console
# ---------------------------------------------------------------------------

def log_metrics(metrics: Dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation metrics."""
    split = metrics.get("split", "?")
    n = metrics.get("num_samples", "?")

    logger.info(f"\n{'=' * 55}")
    logger.info(f" EVALUATION RESULTS — {split.upper()} ({n} samples)")
    logger.info(f"{'=' * 55}")

    emo = metrics["emotion"]
    logger.info(f"\n  EMOTION ({len(EMOTION_LABELS)} classes):")
    logger.info(f"    Accuracy:      {emo['accuracy']:.4f}")
    logger.info(f"    Weighted F1:   {emo['weighted_f1']:.4f}  ← primary metric")
    logger.info(f"    Macro F1:      {emo['macro_f1']:.4f}")
    logger.info(f"    Per-class F1:")
    for label, score in emo["per_class_f1"].items():
        logger.info(f"      {label:10s}: {score:.4f}")

    sent = metrics["sentiment"]
    logger.info(f"\n  SENTIMENT ({len(SENTIMENT_LABELS)} classes):")
    logger.info(f"    Accuracy:      {sent['accuracy']:.4f}")
    logger.info(f"    Weighted F1:   {sent['weighted_f1']:.4f}")
    logger.info(f"    Macro F1:      {sent['macro_f1']:.4f}")
    logger.info(f"    Per-class F1:")
    for label, score in sent["per_class_f1"].items():
        logger.info(f"      {label:10s}: {score:.4f}")

    logger.info(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Full evaluation pipeline (convenience function)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split: str,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation and optionally save results to a JSON file.

    This is the main entry point called from the training loop and notebook.

    Args:
        model:      Trained GatedMultimodalFusion model.
        dataloader: DataLoader for the split to evaluate.
        device:     torch.device.
        split:      'train', 'dev', or 'test'.
        save_path:  Optional path to save metrics.json.

    Returns:
        Metrics dictionary.
    """
    logger.info(f"Running evaluation on: {split}")

    emo_preds, emo_labels, sent_preds, sent_labels = run_evaluation(
        model, dataloader, device
    )

    metrics = compute_metrics(emo_preds, emo_labels, sent_preds, sent_labels, split)
    log_metrics(metrics)

    if save_path:
        save_json(metrics, save_path)
        logger.info(f"Metrics saved: {save_path}")

    return metrics
