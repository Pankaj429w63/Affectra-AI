"""
Affectra AI — Training Loop
=============================
Full training loop for the GatedMultimodalFusion model.

Features:
  - PyTorch mixed precision (AMP) when CUDA is available
  - Gradient clipping to prevent exploding gradients
  - Early stopping based on dev weighted F1
  - ReduceLROnPlateau learning rate scheduler
  - Checkpoint saving after every epoch improvement
  - Training resumption from an existing checkpoint
  - Reproducible random seed
  - Detailed per-epoch logging
  - Clear CUDA OOM guidance (no silent retry)

IMPORTANT — Out of Memory (OOM) errors:
  If you see "CUDA out of memory", reduce BATCH_SIZE in config.py:
    Default: 64
    Try:     32  (first reduction)
    Try:     16  (if still OOM)
  Do NOT reduce batch size automatically — this would hide CUDA errors.
"""

import os
import time
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.src.config import (
    ALPHA_EMOTION,
    BATCH_SIZE,
    BETA_SENTIMENT,
    COLAB_CHECKPOINT_DIR,
    COLAB_LOG_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from training.src.evaluate import evaluate
from training.src.utils import (
    format_metrics,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)

logger = get_logger(__name__, log_dir=COLAB_LOG_DIR)


# ---------------------------------------------------------------------------
# Training State Tracker (Early Stopping)
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Args:
        patience:  Number of epochs without improvement before stopping.
        mode:      'max' if higher metric is better (e.g., F1), 'min' for loss.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, score: float) -> bool:
        """
        Update state and return True if this score is a new best.

        Args:
            score: Current validation metric value.

        Returns:
            True if new best (caller should save checkpoint).
        """
        if self.best_score is None:
            self.best_score = score
            return True  # First epoch is always a "new best"

        if self.mode == "max":
            improved = score >= self.best_score + self.min_delta
        else:
            improved = score <= self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            logger.info(
                f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_score:.4f})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    alpha: float,
    beta: float,
    device: torch.device,
    scaler: Optional[GradScaler],
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Run one training epoch over the DataLoader.

    Args:
        model:               GatedMultimodalFusion in training mode.
        dataloader:          DataLoader for the training split.
        optimizer:           AdamW optimiser.
        emotion_criterion:   Weighted CrossEntropyLoss for emotions.
        sentiment_criterion: Weighted CrossEntropyLoss for sentiments.
        alpha:               Emotion loss weight.
        beta:                Sentiment loss weight.
        device:              torch.device.
        scaler:              GradScaler for AMP, or None for CPU training.
        grad_clip:           Maximum gradient norm (1.0 is standard).

    Returns:
        Dict with 'loss', 'emotion_loss', 'sentiment_loss' averages.
    """
    model.train()

    total_loss      = 0.0
    total_emo_loss  = 0.0
    total_sent_loss = 0.0
    n_batches       = 0

    for batch in dataloader:
        text_feat      = batch["text_feat"].to(device)
        audio_feat     = batch["audio_feat"].to(device)
        video_feat     = batch["video_feat"].to(device)
        text_mask      = batch["text_mask"].to(device)
        audio_mask     = batch["audio_mask"].to(device)
        video_mask     = batch["video_mask"].to(device)
        emotion_label  = batch["emotion_label"].to(device)
        sentiment_label = batch["sentiment_label"].to(device)

        optimizer.zero_grad()

        # ── Forward pass (with optional AMP) ─────────────────────────────
        if scaler is not None:
            with autocast():
                emo_logits, sent_logits = model(
                    text_feat, audio_feat, video_feat,
                    text_mask, audio_mask, video_mask,
                )
                emo_loss  = emotion_criterion(emo_logits, emotion_label)
                sent_loss = sentiment_criterion(sent_logits, sentiment_label)
                loss      = alpha * emo_loss + beta * sent_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

        else:
            # CPU or non-AMP path
            emo_logits, sent_logits = model(
                text_feat, audio_feat, video_feat,
                text_mask, audio_mask, video_mask,
            )
            emo_loss  = emotion_criterion(emo_logits, emotion_label)
            sent_loss = sentiment_criterion(sent_logits, sentiment_label)
            loss      = alpha * emo_loss + beta * sent_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss      += loss.item()
        total_emo_loss  += emo_loss.item()
        total_sent_loss += sent_loss.item()
        n_batches       += 1

    n_batches = max(n_batches, 1)
    return {
        "loss":           total_loss / n_batches,
        "emotion_loss":   total_emo_loss / n_batches,
        "sentiment_loss": total_sent_loss / n_batches,
    }


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    device: torch.device,
    alpha: float = ALPHA_EMOTION,
    beta: float = BETA_SENTIMENT,
    max_epochs: int = MAX_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: str = COLAB_CHECKPOINT_DIR,
    resume_from: Optional[str] = None,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Full training loop with AMP, early stopping, LR scheduling, and checkpointing.

    Args:
        model:               GatedMultimodalFusion model.
        train_loader:        DataLoader for training split.
        dev_loader:          DataLoader for dev split.
        emotion_criterion:   Weighted CrossEntropyLoss for emotions.
        sentiment_criterion: Weighted CrossEntropyLoss for sentiments.
        device:              torch.device.
        alpha:               Emotion loss weight (default from config).
        beta:                Sentiment loss weight (default from config).
        max_epochs:          Maximum training epochs (default from config).
        learning_rate:       Initial learning rate (default from config).
        weight_decay:        AdamW weight decay (default from config).
        patience:            Early stopping patience (default from config).
        checkpoint_dir:      Directory to save checkpoints.
        resume_from:         Path to a checkpoint file to resume training from.
        use_amp:             Use AMP if CUDA is available.

    Returns:
        Training history dictionary with per-epoch metrics.

    Raises:
        RuntimeError: If CUDA OOM is detected (with clear guidance message).
    """
    set_seed(RANDOM_SEED)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True,
    )

    # ── Mixed Precision Scaler ─────────────────────────────────────────────
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler() if amp_enabled else None
    if amp_enabled:
        logger.info("Mixed precision (AMP) enabled.")
    else:
        logger.info("AMP disabled (CPU training or use_amp=False).")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch + 1}.")

    # ── Early stopping tracker ─────────────────────────────────────────────
    early_stop = EarlyStopping(patience=patience, mode="max")
    best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")

    # ── Training history ───────────────────────────────────────────────────
    history = {
        "epochs": [],
        "best_dev_emotion_weighted_f1": 0.0,
        "best_epoch": -1,
    }

    logger.info("=" * 60)
    logger.info(f"Starting training — max {max_epochs} epochs, patience {patience}")
    logger.info(f"  Batch size: {BATCH_SIZE}  |  LR: {learning_rate}  |  AMP: {amp_enabled}")
    logger.info(
        "  ⚠️  If you see 'CUDA out of memory', reduce BATCH_SIZE in config.py "
        "(try 32 first, then 16)."
    )
    logger.info("=" * 60)

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        try:
            train_metrics = train_one_epoch(
                model, train_loader, optimizer,
                emotion_criterion, sentiment_criterion,
                alpha, beta, device, scaler,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "\n\n"
                    "❌ CUDA OUT OF MEMORY during training.\n"
                    "   Fix: reduce BATCH_SIZE in training/src/config.py\n"
                    "   Current: BATCH_SIZE = 64\n"
                    "   Try:     BATCH_SIZE = 32   (reduce to 32 first)\n"
                    "   Try:     BATCH_SIZE = 16   (if 32 still OOMs)\n"
                    "   After changing config.py, re-run the notebook from "
                    "the training cell.\n"
                ) from e
            raise  # Re-raise non-OOM errors unchanged

        # ── Evaluate on dev ────────────────────────────────────────────────
        dev_metrics = evaluate(model, dev_loader, device, split="dev")
        dev_wf1 = dev_metrics["emotion"]["weighted_f1"]

        # ── LR schedule ────────────────────────────────────────────────────
        scheduler.step(dev_wf1)

        # ── Log epoch summary ──────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 4),
            "train_emotion_loss": round(train_metrics["emotion_loss"], 4),
            "train_sentiment_loss": round(train_metrics["sentiment_loss"], 4),
            "dev_emotion_weighted_f1": round(dev_wf1, 4),
            "dev_emotion_macro_f1": round(dev_metrics["emotion"]["macro_f1"], 4),
            "dev_sentiment_weighted_f1": round(dev_metrics["sentiment"]["weighted_f1"], 4),
            "learning_rate": current_lr,
            "elapsed_seconds": round(elapsed, 1),
        }
        history["epochs"].append(epoch_summary)

        logger.info(
            f"Epoch {epoch + 1:3d}/{max_epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"dev_emo_wF1={dev_wf1:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.0f}s"
        )

        # ── Early stopping / checkpoint ────────────────────────────────────
        is_best = early_stop(dev_wf1)
        if is_best:
            history["best_dev_emotion_weighted_f1"] = dev_wf1
            history["best_epoch"] = epoch + 1

            save_checkpoint(
                model, optimizer, epoch,
                metrics=epoch_summary,
                checkpoint_dir=checkpoint_dir,
                filename="checkpoint_best.pt",
            )
            logger.info(f"  🏆 New best model! dev weighted F1 = {dev_wf1:.4f}")

        # Also save latest (for resumption)
        save_checkpoint(
            model, optimizer, epoch,
            metrics=epoch_summary,
            checkpoint_dir=checkpoint_dir,
            filename="checkpoint_latest.pt",
        )

        # ── Save training history ──────────────────────────────────────────
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        save_json(history, history_path)

        if early_stop.should_stop:
            logger.info(
                f"⏹  Early stopping triggered after epoch {epoch + 1}. "
                f"Best dev emotion weighted F1 = {history['best_dev_emotion_weighted_f1']:.4f} "
                f"at epoch {history['best_epoch']}."
            )
            break

    logger.info("Training complete.")
    logger.info(
        f"Best checkpoint: {best_checkpoint_path}  "
        f"(epoch {history['best_epoch']}, "
        f"dev emotion wF1={history['best_dev_emotion_weighted_f1']:.4f})"
    )
    return history
