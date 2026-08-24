"""
Affectra AI — Utility Functions
================================
Shared helpers used across the training pipeline:
  - Logging setup
  - Random seed setting for reproducibility
  - Device detection
  - Phase guard (skip completed pipeline phases)
  - Progress display
  - JSON save/load helpers
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Create and return a logger that writes to both stdout and an optional
    log file. Each call returns the same logger if it was already created.

    Args:
        name:    Logger name (typically the module __name__).
        log_dir: Optional directory to write a timestamped log file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return existing logger
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducible
    results across runs.

    Args:
        seed: Integer seed value. Default matches config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms — may slow down training slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Detect and return the best available PyTorch device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✅ GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found — training will be very slow on CPU.")
        print("    In Google Colab: Runtime → Change runtime type → GPU")
    return device


def check_mixed_precision(device: torch.device) -> bool:
    """
    Return True if automatic mixed precision (AMP) is supported.
    AMP is only beneficial and safe on CUDA devices.

    Args:
        device: The active PyTorch device.

    Returns:
        bool: True if AMP should be used.
    """
    return device.type == "cuda"


# ---------------------------------------------------------------------------
# Phase Guard (Skip Completed Pipeline Phases)
# ---------------------------------------------------------------------------

def phase_guard(phase_name: str, sentinel_path: str) -> bool:
    """
    Skip a pipeline phase if its sentinel output file already exists
    on Google Drive (or local disk). This allows training to resume
    after a Colab session disconnects without redoing work.

    Usage example:
        if phase_guard("Extract text features", "/content/drive/.../text_train.pt"):
            text_feats = torch.load(...)
        else:
            text_feats = extract_text_features(...)
            torch.save(text_feats, ...)

    Args:
        phase_name:    Human-readable name of the phase (for logging).
        sentinel_path: Path to the file that marks this phase complete.

    Returns:
        True  — phase is already complete; caller should skip extraction.
        False — phase has not run yet; caller should execute it.
    """
    if os.path.exists(sentinel_path):
        print(f"✅ Phase '{phase_name}' already complete. Skipping.")
        print(f"   Found: {sentinel_path}")
        return True
    print(f"🔄 Running phase: '{phase_name}' ...")
    return False


# ---------------------------------------------------------------------------
# JSON Helpers
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Serialise a dictionary to a JSON file, creating parent directories
    as needed.

    Args:
        data:   Dictionary to serialise.
        path:   Destination file path.
        indent: JSON indentation level.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"💾 Saved JSON: {path}")


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Checkpoint Save / Load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "checkpoint_latest.pt",
) -> str:
    """
    Save a training checkpoint containing the model state dict, optimiser
    state, epoch number, and current metrics.

    IMPORTANT: We save only state_dicts, never the entire Python model object.
    This avoids pickle issues and keeps checkpoints portable.

    Args:
        model:           The GatedMultimodalFusion model being trained.
        optimizer:       The AdamW optimiser.
        epoch:           Current epoch (0-indexed).
        metrics:         Dictionary of evaluation metrics at this epoch.
        checkpoint_dir:  Directory to save the checkpoint file.
        filename:        Checkpoint filename.

    Returns:
        str: Full path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"💾 Checkpoint saved: {path}  (epoch {epoch + 1})")
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint into an existing model (and optionally
    an optimiser). Used for both training resumption and final evaluation.

    Args:
        path:      Path to the checkpoint file.
        model:     Model instance with the correct architecture.
        optimizer: Optional optimiser to restore state into.
        device:    Device to map tensors onto.

    Returns:
        The raw checkpoint dictionary (contains 'epoch', 'metrics', etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_loc = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", -1)
    metrics = ckpt.get("metrics", {})
    print(f"📂 Checkpoint loaded: {path}  (epoch {epoch + 1})")
    if metrics:
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    return ckpt


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, Any], prefix: str = "") -> str:
    """
    Format a metrics dictionary as a human-readable string for logging.

    Args:
        metrics: Dictionary of metric name → value.
        prefix:  Optional prefix string (e.g., "Epoch 5 dev").

    Returns:
        Formatted string.
    """
    parts = [f"{prefix}:" if prefix else ""]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"  {k}={v:.4f}")
        elif isinstance(v, (int, str, bool)):
            parts.append(f"  {k}={v}")
    return " ".join(parts)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Trainable parameters: {total:,}")
    return total


if __name__ == "__main__":
    # Quick self-test
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Mixed precision: {check_mixed_precision(device)}")
    print(f"Phase guard (missing): {phase_guard('test', '/nonexistent/file.pt')}")
