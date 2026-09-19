import os
import sys
import json

# Add project root to sys.path if not present
sys.path.insert(0, os.path.abspath("."))

import torch
import training.src.train as train_mod
from training.src.train import run_full_training
from training.src.config import CLASS_WEIGHT_POWER

# Monkey-patch evaluate to skip test evaluation entirely
original_evaluate = train_mod.evaluate

def patched_evaluate(model, dataloader, device, split, save_path=None):
    if split == "test":
        print("\n--- SKIPPING TEST EVALUATION AS REQUESTED ---")
        return {"split": "test", "skipped": True}
    return original_evaluate(model, dataloader, device, split, save_path)

train_mod.evaluate = patched_evaluate

print("=======================================================")
print("🚀 STARTING EXPERIMENT 3 SANITY RUN (3 EPOCHS)")
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
    max_epochs=3,
    export_artifacts=False
)

print("\n=======================================================")
print("🏁 EXPERIMENT 3 SANITY RUN FINAL REPORT")
print("=======================================================")
best_epoch = results["history"]["best_epoch"]
best_wf1 = results["history"]["best_dev_emotion_weighted_f1"]

print(f"Best Epoch: {best_epoch}")
print(f"Best Dev Emotion Weighted F1:   {best_wf1:.4f}")

epochs_completed = len(results["history"]["epochs"])
print("-------------------------------------------------------")
for ep in results["history"]["epochs"]:
    print(f"Epoch {ep['epoch']}: Train Loss = {ep['train_loss']:.4f} | Train Emo wF1 = {ep['train_emotion_weighted_f1']:.4f} | Train Emo mF1 = {ep['train_emotion_macro_f1']:.4f} | Dev Emo wF1 = {ep['dev_emotion_weighted_f1']:.4f} | Dev Emo mF1 = {ep['dev_emotion_macro_f1']:.4f} | Dev Sent wF1 = {ep['dev_sentiment_weighted_f1']:.4f} | Gap = {ep['generalization_gap_wf1']:+.4f} | LR = {ep['learning_rate']:.2e}")
print("-------------------------------------------------------")
print(f"Number of Epochs Completed:      {epochs_completed}")
print(f"Exact Path of Best Checkpoint:   {os.path.abspath('data/exp3_checkpoints/checkpoint_best.pt')}")
print("Test Set Bypassed:               Yes")
print("Run Completed Successfully:      Yes")
print("=======================================================\n")
