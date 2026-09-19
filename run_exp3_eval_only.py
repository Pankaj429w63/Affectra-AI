import os
import sys
import json

# Add project root to sys.path if not present
sys.path.insert(0, os.path.abspath("."))

import torch
import training.src.train as train_mod
from training.src.fusion_model import build_model
from training.src.evaluate import evaluate
from training.src.dataset import MELDCachedDataset
from torch.utils.data import DataLoader

print("\n--- RUNNING FINAL DEV EVALUATION ON BEST CHECKPOINT ---")
# Build and load the best model
device = torch.device("cpu")
model = build_model(device=device)

train_mod.load_checkpoint(os.path.join("data/exp3_checkpoints", "checkpoint_best.pt"), model, device=device)

# Load Dev dataset
train_text, dev_text, test_text = train_mod.load_all_splits("text", "data/feature_cache")
train_audio, dev_audio, test_audio = train_mod.load_all_splits("audio", "data/feature_cache")
train_video, dev_video, test_video = train_mod.load_all_splits("video", "data/feature_cache")

train_df, dev_df, test_df = train_mod.load_all_metadata("data/MELD/annotations")
dev_ds = MELDCachedDataset(dev_df, dev_text, dev_audio, dev_video)
dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False)

best_dev_metrics = evaluate(model, dev_loader, device, split="dev")

print("\n=======================================================")
print("EXPERIMENT 3 SUMMARY")
print("=======================================================")

emo_metrics = best_dev_metrics["emotion"]
per_class_f1 = emo_metrics.get("per_class_f1", {})
sent_metrics = best_dev_metrics["sentiment"]

# From the history output:
best_epoch = 19
train_emo_wf1 = 0.5486
gap = 0.0222

print(f"Best epoch: {best_epoch}")
print(f"Best dev emotion weighted F1: {emo_metrics['weighted_f1'] * 100:.2f}%")
print(f"Best dev emotion macro F1: {emo_metrics['macro_f1'] * 100:.2f}%")
print(f"Best dev sentiment weighted F1: {sent_metrics['weighted_f1'] * 100:.2f}%")

print(f"Train emotion weighted F1 at best epoch: {train_emo_wf1 * 100:.2f}%")
print(f"Generalization gap: {gap * 100:+.2f}%")

print(f"Disgust F1: {per_class_f1.get('disgust', 0) * 100:.2f}%")
print(f"Fear F1: {per_class_f1.get('fear', 0) * 100:.2f}%")
print(f"Anger F1: {per_class_f1.get('anger', 0) * 100:.2f}%")
print(f"Sadness F1: {per_class_f1.get('sadness', 0) * 100:.2f}%")
print(f"Joy F1: {per_class_f1.get('joy', 0) * 100:.2f}%")
print(f"Surprise F1: {per_class_f1.get('surprise', 0) * 100:.2f}%")
print(f"Neutral F1: {per_class_f1.get('neutral', 0) * 100:.2f}%")

print("\nThen compare against Experiment 2:")
print("Experiment 2:")
print("Dev weighted F1 = 52.63%")
print("Dev macro F1 = 35.07%")
print("\nExperiment 3:")
print(f"Dev weighted F1 = {emo_metrics['weighted_f1'] * 100:.2f}%")
print(f"Dev macro F1 = {emo_metrics['macro_f1'] * 100:.2f}%")

