import os
import sys
import json
import torch
import torch.nn as nn
from collections import Counter


# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from training.src.config import (
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    DROPOUT, MODALITY_DROPOUT, CLASS_WEIGHT_POWER, LABEL_SMOOTHING
)
from training.src.dataset import load_meld_metadata, MELDCachedDataset, build_dataloader
from training.src.feature_cache import load_all_splits
from training.src.train import train, evaluate, load_checkpoint
from training.src.fusion_model import ModalityProjection, build_weighted_loss, count_parameters

# ===========================================================================
# 1. EXPLICIT DEDICATED MODEL FOR EXP 5A (NO VIDEO AT ALL)
# ===========================================================================

class GatedTextAudioFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        fusion_dim: int = 512,
        emotion_classes: int = 7,
        sentiment_classes: int = 3,
        dropout: float = DROPOUT,
        modality_dropout: float = MODALITY_DROPOUT,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout

        # Projections (Text + Audio ONLY)
        self.text_proj = ModalityProjection(text_dim, fusion_dim, dropout)
        self.audio_proj = ModalityProjection(audio_dim, fusion_dim, dropout)

        # Contextual Gating (Text + Audio ONLY)
        self.text_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        # Cross-Modal Interaction (2 modalities = 2 * fusion_dim)
        self.cross_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # Fusion Regularization
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # Task Heads
        self.emotion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, emotion_classes),
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, sentiment_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_modality_dropout(self, t_mask: torch.Tensor, a_mask: torch.Tensor) -> tuple:
        if not self.training or self.modality_dropout_prob <= 0.0:
            return t_mask, a_mask

        device = t_mask.device
        b = t_mask.size(0)

        t_keep = (torch.rand(b, 1, device=device) > self.modality_dropout_prob).float()
        a_keep = (torch.rand(b, 1, device=device) > self.modality_dropout_prob).float()

        new_t = t_mask * t_keep
        new_a = a_mask * a_keep

        # Guard: if all modalities were dropped for a sample, revert to original mask
        all_dropped = (new_t + new_a) < 0.5
        new_t = torch.where(all_dropped, t_mask, new_t)
        new_a = torch.where(all_dropped, a_mask, new_a)

        return new_t, new_a

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> tuple:
        t_mask = text_mask.view(-1, 1).float()
        a_mask = audio_mask.view(-1, 1).float()

        t_mask, a_mask = self._apply_modality_dropout(t_mask, a_mask)

        t_proj = self.text_proj(text_feat, t_mask)
        a_proj = self.audio_proj(audio_feat, a_mask)

        active_counts = (t_mask + a_mask).clamp(min=1.0)
        context = (t_proj + a_proj) / active_counts

        t_in = torch.cat([t_proj, context], dim=-1)
        a_in = torch.cat([a_proj, context], dim=-1)

        t_gate = torch.sigmoid(self.text_gate(t_in)) * t_mask
        a_gate = torch.sigmoid(self.audio_gate(a_in)) * a_mask

        gate_sum = (t_gate + a_gate).clamp(min=1e-6)
        t_weight = t_gate / gate_sum
        a_weight = a_gate / gate_sum

        gated_sum = t_weight * t_proj + a_weight * a_proj

        all_modalities = torch.cat([t_proj, a_proj], dim=-1)
        cross_inter = self.cross_proj(all_modalities)

        fused = gated_sum + cross_inter + context
        fused = self.fusion_norm(self.fusion_dropout(fused))

        emotion_logits = self.emotion_head(fused)
        sentiment_logits = self.sentiment_head(fused)

        return emotion_logits, sentiment_logits

# ===========================================================================
# 2. EXPERIMENT RUNNER
# ===========================================================================

def run_exp5a_training(epochs: int):
    print("==================================================")
    print("1. VERIFY MODEL STRUCTURE BEFORE TRAINING")
    print("==================================================")
    
    device = torch.device("cpu")
    model = GatedTextAudioFusion().to(device)
    
    param_count = count_parameters(model)
    print(f"Model Parameters: {param_count:,}")
    
    print("\nState Dict Keys:")
    contains_video = False
    for key in model.state_dict().keys():
        print("  - " + key)
        if "video" in key:
            contains_video = True
            
    print(f"\nContains video parameters: {'YES' if contains_video else 'NO'}")
    if contains_video:
        print("FATAL ERROR: Video parameters found in model! Stopping.")
        sys.exit(1)
        
    print("\nRunning dummy forward pass...")
    b_size = 4
    dummy_text = torch.randn(b_size, 768)
    dummy_audio = torch.randn(b_size, 768)
    dummy_t_mask = torch.ones(b_size)
    dummy_a_mask = torch.ones(b_size)
    dummy_v_feat = torch.zeros(b_size, 768)
    dummy_v_mask = torch.zeros(b_size)
    e_out, s_out = model(dummy_text, dummy_audio, dummy_v_feat, dummy_t_mask, dummy_a_mask, dummy_v_mask)
    print(f"Emotion output shape: {e_out.shape} (expected [{b_size}, 7])")
    print(f"Sentiment output shape: {s_out.shape} (expected [{b_size}, 3])")
    
    assert e_out.shape == (b_size, 7)
    assert s_out.shape == (b_size, 3)
    print("✅ Model verification passed.")
    
    print("\n==================================================")
    print("2. LOAD DATA")
    print("==================================================")
    
    cache_dir = "data/feature_cache"
    checkpoint_dir = "data/exp5a_correct_checkpoints"
    output_dir = "data/exp5a_correct_output"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = load_meld_metadata("data/MELD/annotations/train_sent_emo.csv")
    dev_df = load_meld_metadata("data/MELD/annotations/dev_sent_emo.csv")
    
    train_text, dev_text, _ = load_all_splits("text", cache_dir)
    train_audio, dev_audio, _ = load_all_splits("audio", cache_dir)
    
    print("\n==================================================")
    print(f"6. {epochs}-EPOCH FULL RUN (EXP 5A CORRECT)")
    print("==================================================")
    print("Experiment: Experiment 5A CORRECT")
    print("Architecture: GatedTextAudioFusion")
    print("Active modalities: Text = YES, Audio = YES, Video = NO")
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Text feature shape: {train_text.shape} / {dev_text.shape}")
    print(f"Audio feature shape: {train_audio.shape} / {dev_audio.shape}")
    print("Video features loaded: NO")
    print("WeightedRandomSampler: NO")
    print(f"Class weighting: YES, CLASS_WEIGHT_POWER={CLASS_WEIGHT_POWER}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Parameter count: {param_count:,}")
    print("==================================================\n")
    
    # We pass None for video features, MELDCachedDataset handles it by setting mask=0
    train_ds = MELDCachedDataset(train_df, train_text, train_audio, None)
    dev_ds = MELDCachedDataset(dev_df, dev_text, dev_audio, None)
    
    train_loader = build_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
    dev_loader = build_dataloader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    
    emo_counts = dict(Counter(train_df["emotion_norm"].tolist()))
    sent_counts = dict(Counter(train_df["sentiment_norm"].tolist()))
    
    # Build loss
    emo_crit, sent_crit, alpha, beta = build_weighted_loss(
        emo_counts, sent_counts, device,
        emotion_weight=0.6, sentiment_weight=0.4,
        power=CLASS_WEIGHT_POWER, label_smoothing=LABEL_SMOOTHING
    )
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        emotion_criterion=emo_crit,
        sentiment_criterion=sent_crit,
        device=device,
        alpha=alpha,
        beta=beta,
        max_epochs=epochs,
        learning_rate=LEARNING_RATE,
        patience=EARLY_STOPPING_PATIENCE,
        checkpoint_dir=checkpoint_dir,
    )
    
    print("\n=======================================================")
    print(f"🏁 EXPERIMENT 5A CORRECT ({epochs} EPOCHS) FINAL REPORT")
    print("=======================================================")
    best_epoch = history["best_epoch"]
    best_wf1 = history["best_dev_emotion_weighted_f1"]

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Dev Emotion Weighted F1:   {best_wf1:.4f}")

    epochs_completed = len(history["epochs"])
    print("-------------------------------------------------------")
    for ep in history["epochs"]:
        print(f"Epoch {ep['epoch']}: Train Loss = {ep['train_loss']:.4f} | Train Emo wF1 = {ep['train_emotion_weighted_f1']:.4f} | Train Emo mF1 = {ep['train_emotion_macro_f1']:.4f} | Dev Emo wF1 = {ep['dev_emotion_weighted_f1']:.4f} | Dev Emo mF1 = {ep['dev_emotion_macro_f1']:.4f} | Dev Sent wF1 = {ep['dev_sentiment_weighted_f1']:.4f} | Gap = {ep['generalization_gap_wf1']:+.4f} | LR = {ep['learning_rate']:.2e}")
    print("-------------------------------------------------------")
    print(f"Number of Epochs Completed:      {epochs_completed}")
    print(f"Exact Path of Best Checkpoint:   {os.path.abspath(os.path.join(checkpoint_dir, 'checkpoint_best.pt'))}")
    print("Test Set Bypassed:               Yes")
    print("Run Completed Successfully:      Yes")
    print("=======================================================\n")

if __name__ == "__main__":
    run_exp5a_training(epochs=30)
