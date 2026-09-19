import os
import sys

# Add project root to sys.path if not present
sys.path.insert(0, os.path.abspath("."))

import torch
import training.src.train as train_mod
from training.src.train import run_full_training

# Monkey-patch evaluate to skip test evaluation entirely
original_evaluate = train_mod.evaluate

def patched_evaluate(model, dataloader, device, split, save_path=None):
    if split == "test":
        print("\n--- SKIPPING TEST EVALUATION AS REQUESTED ---")
        return {"split": "test", "skipped": True}
    return original_evaluate(model, dataloader, device, split, save_path)

train_mod.evaluate = patched_evaluate

# (Removed metadata patch to allow MELDCachedDataset length validation to pass)


print("=======================================================")
print("🚀 STARTING FULL EXPERIMENT 1 CPU TRAINING")
print("=======================================================")
print("Constraints applied:")
print("- CPU Only")
print("- No Test Set Evaluation")
print("- Separate Checkpoint Directory: data/exp1_checkpoints")
print("- Separate Output Directory: data/exp1_output")
print("=======================================================\n")

results = run_full_training(
    checkpoint_dir="data/exp1_checkpoints",
    output_dir="data/exp1_output",
    max_epochs=30,
    export_artifacts=False
)

print("\n=======================================================")
print("🏁 EXPERIMENT 1 FINAL REPORT")
print("=======================================================")
best_epoch = results["history"]["best_epoch"]
best_wf1 = results["history"]["best_dev_emotion_weighted_f1"]

best_metrics = None
for ep in results["history"]["epochs"]:
    if ep["epoch"] == best_epoch:
        best_metrics = ep
        break

if best_metrics:
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Dev Emotion Weighted F1:   {best_wf1:.4f}")
    print(f"Best Dev Emotion Macro F1:      {best_metrics['dev_emotion_macro_f1']:.4f}")
    print(f"Best Dev Sentiment Weighted F1: {best_metrics['dev_sentiment_weighted_f1']:.4f}")
    print("-------------------------------------------------------")
    print(f"Corresponding Train Emotion wF1: {best_metrics['train_emotion_weighted_f1']:.4f}")
    print(f"Corresponding Train Emotion mF1: {best_metrics['train_emotion_macro_f1']:.4f}")
    print(f"Final Train/Dev Gap (wF1):       {best_metrics['generalization_gap_wf1']:+.4f}")
    print(f"Final Learning Rate:             {best_metrics['learning_rate']:.2e}")
else:
    print("Could not retrieve best epoch details.")

epochs_completed = len(results["history"]["epochs"])
print("-------------------------------------------------------")
print(f"Number of Epochs Completed:      {epochs_completed}")
print(f"Early Stopping Occurred:         {'Yes' if epochs_completed < 30 else 'No'}")
print(f"Exact Path of Best Checkpoint:   {os.path.abspath('data/exp1_checkpoints/checkpoint_best.pt')}")
print("Run Completed Successfully:      Yes")
print("=======================================================\n")
