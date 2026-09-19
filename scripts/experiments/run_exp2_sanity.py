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

print("=======================================================")
print("🚀 STARTING EXPERIMENT 2 SANITY RUN (3 EPOCHS)")
print("=======================================================")
print("Constraints applied:")
print("- CPU Only")
print("- No Test Set Evaluation")
print("- Separate Checkpoint Directory: data/exp2_checkpoints")
print("- Separate Output Directory: data/exp2_output")
print("=======================================================\n")

results = run_full_training(
    checkpoint_dir="data/exp2_checkpoints",
    output_dir="data/exp2_output",
    max_epochs=3,
    export_artifacts=False
)

print("\n=======================================================")
print("🏁 EXPERIMENT 2 SANITY RUN FINAL REPORT")
print("=======================================================")
best_epoch = results["history"]["best_epoch"]
best_wf1 = results["history"]["best_dev_emotion_weighted_f1"]

print(f"Best Epoch: {best_epoch}")
print(f"Best Dev Emotion Weighted F1:   {best_wf1:.4f}")

epochs_completed = len(results["history"]["epochs"])
print("-------------------------------------------------------")
print(f"Number of Epochs Completed:      {epochs_completed}")
print(f"Exact Path of Best Checkpoint:   {os.path.abspath('data/exp2_checkpoints/checkpoint_best.pt')}")
print("Test Set Bypassed:               Yes")
print("Run Completed Successfully:      Yes")
print("=======================================================\n")
