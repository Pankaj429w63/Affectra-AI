import os
import sys
import json

# Add project root to sys.path if not present
sys.path.insert(0, os.path.abspath("."))

import torch
import training.src.train as train_mod
from training.src.train import run_full_training
from training.src.config import CLASS_WEIGHT_POWER
from training.src.evaluate import evaluate
from training.src.dataset import MELDCachedDataset
from torch.utils.data import DataLoader

# Monkey-patch evaluate to skip test evaluation entirely
original_evaluate = train_mod.evaluate

def patched_evaluate(model, dataloader, device, split, save_path=None):
    if split == "test":
        print("\n--- SKIPPING TEST EVALUATION AS REQUESTED ---")
        return {"split": "test", "skipped": True}
    return original_evaluate(model, dataloader, device, split, save_path)

train_mod.evaluate = patched_evaluate

print("=======================================================")
print("🚀 STARTING FULL EXPERIMENT 3 CPU TRAINING")
print("=======================================================")
print("Constraints applied:")
print("- CPU Only")
print("- No Test Set Evaluation")
print("- Separate Checkpoint Directory: data/exp3_checkpoints")
print("- Separate Output Directory: data/exp3_output")
print(f"- VERIFIED: CLASS_WEIGHT_POWER = {CLASS_WEIGHT_POWER}")
print("=======================================================\n")

results = run_full_training(
    checkpoint_dir="data/exp3_checkpoints",
    output_dir="data/exp3_output",
    max_epochs=30,
    export_artifacts=False
)

print("\n--- RUNNING FINAL DEV EVALUATION ON BEST CHECKPOINT ---")
# Build and load the best model
device = torch.device("cpu")
model = train_mod.GatedMultimodalFusion(
    text_dim=768, audio_dim=768, video_dim=768,
    fusion_dim=train_mod.FUSION_DIM,
    emotion_classes=7, sentiment_classes=3,
    dropout=train_mod.DROPOUT,
    modality_dropout=train_mod.MODALITY_DROPOUT
).to(device)

train_mod.load_checkpoint(os.path.join("data/exp3_checkpoints", "checkpoint_best.pt"), model, device)

# Load Dev dataset
train_text, dev_text, test_text = train_mod.load_all_splits("text", "data/feature_cache")
train_audio, dev_audio, test_audio = train_mod.load_all_splits("audio", "data/feature_cache")
train_video, dev_video, test_video = train_mod.load_all_splits("video", "data/feature_cache")

train_df, dev_df, test_df = train_mod.load_all_metadata("data/MELD/annotations")
dev_ds = MELDCachedDataset(dev_df, dev_text, dev_audio, dev_video)
dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False)

best_dev_metrics = evaluate(model, dev_loader, device, split="dev")
with open("data/exp3_output/best_dev_metrics.json", "w") as f:
    json.dump(best_dev_metrics, f, indent=4)

print("\n=======================================================")
print("EXPERIMENT 3 SUMMARY")
print("=======================================================")

best_epoch = results["history"]["best_epoch"]
history = results["history"]["epochs"]

best_history_metrics = None
for ep in history:
    if ep["epoch"] == best_epoch:
        best_history_metrics = ep
        break

emo_metrics = best_dev_metrics["emotion"]
per_class_f1 = emo_metrics.get("per_class_f1", {})
sent_metrics = best_dev_metrics["sentiment"]

print(f"Best epoch: {best_epoch}")
print(f"Best dev emotion weighted F1: {emo_metrics['weighted_f1'] * 100:.2f}%")
print(f"Best dev emotion macro F1: {emo_metrics['macro_f1'] * 100:.2f}%")
print(f"Best dev sentiment weighted F1: {sent_metrics['weighted_f1'] * 100:.2f}%")

if best_history_metrics:
    print(f"Train emotion weighted F1 at best epoch: {best_history_metrics['train_emotion_weighted_f1'] * 100:.2f}%")
    print(f"Generalization gap: {best_history_metrics['generalization_gap_wf1'] * 100:+.2f}%")

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
