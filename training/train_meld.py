"""
Affectra AI — Automated MELD Training Runner
==============================================
Single-command CLI script to train, evaluate, checkpoint, and export
the Affectra AI multimodal emotion model.

Usage:
    python -m training.train_meld
    python -m training.train_meld --epochs 30 --batch-size 64
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.src.config import (
    BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_OUTPUT_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_ARTIFACT_DIR,
)
from training.src.train import run_full_training
from training.src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Affectra AI Multimodal Fusion Model on MELD")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Max training epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help=f"Early stopping patience (default: {EARLY_STOPPING_PATIENCE})")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Feature cache directory")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint output directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Metrics and reports output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint_latest.pt to resume")
    parser.add_argument("--no-export", action="store_true", help="Skip model artifact export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("\n" + "=" * 65)
    print(" 🧠 AFFECTRA AI — MULTIMODAL FUSION TRAINING PIPELINE")
    print("=" * 65)
    print(f"Device:         {device}")
    print(f"Max Epochs:     {args.epochs}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Patience:       {args.patience}")
    print(f"Cache Dir:      {args.cache_dir}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Export Dir:     {MODEL_ARTIFACT_DIR}")
    print("=" * 65 + "\n")

    results = run_full_training(
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        resume_from=args.resume_from,
        export_artifacts=not args.no_export,
    )

    best_epoch = results["history"]["best_epoch"]
    best_f1 = results["history"]["best_dev_emotion_weighted_f1"]
    test_f1 = results["test_metrics"]["emotion"]["weighted_f1"]
    test_acc = results["test_metrics"]["emotion"]["accuracy"]
    sent_f1 = results["test_metrics"]["sentiment"]["weighted_f1"]

    print("\n" + "=" * 65)
    print(" 🏁 TRAINING & EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Best Dev Epoch:                {best_epoch}")
    print(f"Best Dev Emotion Weighted F1:  {best_f1:.4f}")
    print(f"Final Test Emotion Weighted F1:{test_f1:.4f}")
    print(f"Final Test Emotion Accuracy:   {test_acc:.4f}")
    print(f"Final Test Sentiment F1:       {sent_f1:.4f}")
    print(f"Model Artifacts Exported to:   {MODEL_ARTIFACT_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
