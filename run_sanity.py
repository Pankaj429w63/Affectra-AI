import os
import sys

# Add project root to sys.path if not present
sys.path.insert(0, os.path.abspath("."))

import torch
import training.src.train as train_mod
from training.src.train import run_full_training

# Monkey-patch train_one_epoch to print mask statistics on the first batch
original_train_one_epoch = train_mod.train_one_epoch
first_batch_checked = False

def patched_train_one_epoch(model, dataloader, optimizer, emotion_criterion, sentiment_criterion, alpha, beta, device, scaler, grad_clip=1.0):
    global first_batch_checked
    if not first_batch_checked:
        for batch in dataloader:
            print(f"\n--- RUNTIME MASK VERIFICATION ---")
            print(f"Text mask mean (active=1.0):  {batch['text_mask'].float().mean().item():.2f}")
            print(f"Audio mask mean (active=1.0): {batch['audio_mask'].float().mean().item():.2f}")
            print(f"Video mask mean (active=1.0): {batch['video_mask'].float().mean().item():.2f}")
            print(f"---------------------------------\n")
            break
        first_batch_checked = True
    
    return original_train_one_epoch(model, dataloader, optimizer, emotion_criterion, sentiment_criterion, alpha, beta, device, scaler, grad_clip)

train_mod.train_one_epoch = patched_train_one_epoch

print("Starting 2-epoch CPU sanity run...")
results = run_full_training(
    checkpoint_dir="data/sanity_checkpoints",
    output_dir="data/sanity_output",
    max_epochs=2,
    export_artifacts=False
)

print("\n--- SANITY RUN METRICS ---")
for epoch_data in results["history"]["epochs"]:
    print(f"Epoch {epoch_data['epoch']}:")
    print(f"  Train Loss: {epoch_data['train_loss']:.4f}")
    print(f"  Train Emotion F1 (W): {epoch_data['train_emotion_weighted_f1']:.4f}")
    print(f"  Train Emotion F1 (M): {epoch_data['train_emotion_macro_f1']:.4f}")
    print(f"  Dev Loss (Not explicitly saved in dict but dev F1 is below)")
    print(f"  Dev Emotion F1 (W):   {epoch_data['dev_emotion_weighted_f1']:.4f}")
    print(f"  Dev Emotion F1 (M):   {epoch_data['dev_emotion_macro_f1']:.4f}")
    print(f"  Generalization Gap:   {epoch_data['generalization_gap_wf1']:+.4f}")
    print(f"  Train Sentiment F1 (W): {epoch_data['train_sentiment_weighted_f1']:.4f}")
    print(f"  Dev Sentiment F1 (W):   {epoch_data['dev_sentiment_weighted_f1']:.4f}")
    print(f"  Learning Rate: {epoch_data['learning_rate']:.2e}")

print("\nSanity run completed successfully.")
