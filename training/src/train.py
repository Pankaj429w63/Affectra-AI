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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
    def get_autocast(device_type="cuda"):
        return autocast(device_type=device_type)
    def get_scaler(device_type="cuda"):
        return GradScaler(device_type)
except (ImportError, TypeError):
    from torch.cuda.amp import GradScaler, autocast
    def get_autocast(device_type="cuda"):
        return autocast()
    def get_scaler(device_type="cuda"):
        return GradScaler()

from training.src.config import (
    EMOTION_LOSS_WEIGHT,
    BATCH_SIZE,
    SENTIMENT_LOSS_WEIGHT,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from training.src.evaluate import evaluate, compute_metrics
from training.src.utils import (
    format_metrics,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)

logger = get_logger(__name__, log_dir=DEFAULT_LOG_DIR)


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

    all_emo_preds = []
    all_emo_labels = []
    all_sent_preds = []
    all_sent_labels = []

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
            with get_autocast(device.type):
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

        emo_preds = emo_logits.detach().argmax(dim=1).cpu().tolist()
        sent_preds = sent_logits.detach().argmax(dim=1).cpu().tolist()
        all_emo_preds.extend(emo_preds)
        all_emo_labels.extend(emotion_label.cpu().tolist())
        all_sent_preds.extend(sent_preds)
        all_sent_labels.extend(sentiment_label.cpu().tolist())

        total_loss      += loss.item()
        total_emo_loss  += emo_loss.item()
        total_sent_loss += sent_loss.item()
        n_batches       += 1

    n_batches = max(n_batches, 1)
    
    # Compute full metrics on training data
    train_metrics = compute_metrics(
        all_emo_preds, all_emo_labels,
        all_sent_preds, all_sent_labels,
        split="train"
    )
    
    return {
        "loss":           total_loss / n_batches,
        "emotion_loss":   total_emo_loss / n_batches,
        "sentiment_loss": total_sent_loss / n_batches,
        "metrics":        train_metrics,
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
    alpha: float = EMOTION_LOSS_WEIGHT,
    beta: float = SENTIMENT_LOSS_WEIGHT,
    max_epochs: int = MAX_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
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

    # ── LR Scheduler (verbose removed for PyTorch 2.2+ compatibility) ───────
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
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
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(dev_wf1)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            logger.info(f"  📉 ReduceLROnPlateau: LR reduced from {old_lr:.2e} to {new_lr:.2e}")

        # ── Log epoch summary ──────────────────────────────────────────────
        elapsed = time.time() - epoch_start
        current_lr = new_lr

        train_wf1 = train_metrics["metrics"]["emotion"]["weighted_f1"]
        train_mf1 = train_metrics["metrics"]["emotion"]["macro_f1"]
        train_sent_wf1 = train_metrics["metrics"]["sentiment"]["weighted_f1"]

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics["loss"], 4),
            "train_emotion_loss": round(train_metrics["emotion_loss"], 4),
            "train_sentiment_loss": round(train_metrics["sentiment_loss"], 4),
            "train_emotion_weighted_f1": round(train_wf1, 4),
            "train_emotion_macro_f1": round(train_mf1, 4),
            "train_sentiment_weighted_f1": round(train_sent_wf1, 4),
            "dev_emotion_weighted_f1": round(dev_wf1, 4),
            "dev_emotion_macro_f1": round(dev_metrics["emotion"]["macro_f1"], 4),
            "dev_sentiment_weighted_f1": round(dev_metrics["sentiment"]["weighted_f1"], 4),
            "generalization_gap_wf1": round(train_wf1 - dev_wf1, 4),
            "learning_rate": current_lr,
            "elapsed_seconds": round(elapsed, 1),
        }
        history["epochs"].append(epoch_summary)

        logger.info(
            f"Epoch {epoch + 1:3d}/{max_epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"train_emo_wF1={train_wf1:.4f} | "
            f"dev_emo_wF1={dev_wf1:.4f} | "
            f"gap={train_wf1 - dev_wf1:+.4f} | "
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


def run_full_training(
    cache_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    resume_from: Optional[str] = None,
    export_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Automated full training workflow:
      1. Resolves MELD annotation CSVs (auto-discovering from local data or Colab)
      2. Loads cached multimodal features (or extracts text features if not yet cached)
      3. Builds class-weighted multi-task loss with power smoothing and label smoothing
      4. Trains GatedMultimodalFusion with ReduceLROnPlateau, EarlyStopping, and checkpointing
      5. Runs final test evaluation on held-out test split
      6. Exports all model artifacts to MODEL_ARTIFACT_DIR
    """
    from collections import Counter
    from training.src.config import (
        DEFAULT_CACHE_DIR,
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_OUTPUT_DIR,
        MELD_DEV_CSV,
        MELD_TEST_CSV,
        MELD_TRAIN_CSV,
        MODEL_ARTIFACT_DIR,
    )
    from training.src.dataset import MELDCachedDataset, build_dataloader, load_meld_metadata
    from training.src.export_model import export_all
    from training.src.feature_cache import load_all_splits, save_features
    from training.src.feature_extractors import TextExtractor
    from training.src.fusion_model import build_model, build_weighted_loss
    from training.src.utils import get_device

    # Force CPU for controlled experiments
    device = torch.device("cpu")
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Locate annotation CSVs ─────────────────────────────────────────
    # Search common candidate locations
    candidates = [
        os.path.join("data", "MELD", "annotations"),
        os.path.join("data", "meld_csv"),
        os.path.join("meld_data", "MELD.Raw"),
        "/content/meld_data/MELD.Raw",
        "/content/MELD_annotations/data/MELD",
        os.path.join("..", "meld_data", "MELD.Raw"),
    ]
    csv_root = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, MELD_TRAIN_CSV)):
            csv_root = cand
            break

    if csv_root is None:
        raise FileNotFoundError(
            f"Could not locate MELD CSVs in any of: {candidates}. "
            "Please ensure train_sent_emo.csv is downloaded."
        )

    logger.info(f"Loading metadata from: {csv_root}")
    train_df = load_meld_metadata(os.path.join(csv_root, MELD_TRAIN_CSV))
    dev_df   = load_meld_metadata(os.path.join(csv_root, MELD_DEV_CSV))
    test_df  = load_meld_metadata(os.path.join(csv_root, MELD_TEST_CSV))

    # ── 2. Load or extract features ───────────────────────────────────────
    logger.info(f"Checking feature cache in: {cache_dir}")
    train_text, dev_text, test_text = load_all_splits("text", cache_dir)
    train_audio, dev_audio, test_audio = load_all_splits("audio", cache_dir)
    train_video, dev_video, test_video = load_all_splits("video", cache_dir)

    # If text features are not cached, extract and save them automatically
    if train_text is None or dev_text is None or test_text is None:
        logger.info("Extracting text features via DistilRoBERTa...")
        text_extractor = TextExtractor(device=device, batch_size=32)
        if train_text is None:
            train_text = text_extractor.extract_batch(train_df["Utterance"].tolist())
            save_features(train_text, "text", "train", cache_dir=cache_dir, sample_ids=train_df["sample_id"].tolist())
        if dev_text is None:
            dev_text = text_extractor.extract_batch(dev_df["Utterance"].tolist())
            save_features(dev_text, "text", "dev", cache_dir=cache_dir, sample_ids=dev_df["sample_id"].tolist())
        if test_text is None:
            test_text = text_extractor.extract_batch(test_df["Utterance"].tolist())
            save_features(test_text, "text", "test", cache_dir=cache_dir, sample_ids=test_df["sample_id"].tolist())
        del text_extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Building PyTorch DataLoaders...")
    train_ds = MELDCachedDataset(train_df, train_text, train_audio, train_video)
    dev_ds   = MELDCachedDataset(dev_df, dev_text, dev_audio, dev_video)
    test_ds  = MELDCachedDataset(test_df, test_text, test_audio, test_video)

    pin_mem = (device.type == "cuda")

    train_loader = build_dataloader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    dev_loader   = build_dataloader(dev_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader  = build_dataloader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    # ── 3. Model and Loss ─────────────────────────────────────────────────
    model = build_model(device)
    emo_counts  = dict(Counter(train_df["emotion_norm"].tolist()))
    sent_counts = dict(Counter(train_df["sentiment_norm"].tolist()))
    emo_crit, sent_crit, alpha, beta = build_weighted_loss(emo_counts, sent_counts, device)

    # ── 4. Train ──────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        emotion_criterion=emo_crit,
        sentiment_criterion=sent_crit,
        device=device,
        alpha=alpha,
        beta=beta,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )

    # ── 5. Final Evaluation on Best Checkpoint ────────────────────────────
    best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model, device=device)

    logger.info("\nRunning final test evaluation...")
    test_metrics = evaluate(
        model, test_loader, device, split="test",
        save_path=os.path.join(output_dir, "test_metrics.json"),
    )

    # ── 6. Export Artifacts ───────────────────────────────────────────────
    if export_artifacts:
        export_dir = MODEL_ARTIFACT_DIR
        logger.info(f"\nExporting model artifacts to: {export_dir}")
        export_all(
            model=model,
            test_metrics=test_metrics,
            output_dir=export_dir,
            also_save_to_drive=os.path.exists("/content/drive"),
        )

    return {
        "history": history,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    run_full_training()
