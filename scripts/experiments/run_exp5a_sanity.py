import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from training.src.fusion_model import ModalityProjection
import training.src.train as train_mod
from training.src.train import run_full_training
from training.src.config import CLASS_WEIGHT_POWER, LABEL_SMOOTHING

# ---------------------------------------------------------------------------
# Experiment 5A Architecture (No Video, completely isolated gates/cross proj)
# ---------------------------------------------------------------------------
class GatedTextAudioFusion(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        fusion_dim: int = 512,
        emotion_classes: int = 7,
        sentiment_classes: int = 3,
        dropout: float = 0.3,
        modality_dropout: float = 0.05,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout

        # ── Modality-Specific Encoders ────────────────────────────────────────
        self.text_proj = ModalityProjection(text_dim, fusion_dim, dropout)
        self.audio_proj = ModalityProjection(audio_dim, fusion_dim, dropout)

        # ── Contextual Gating Mechanism (2 Modalities Only) ───────────────────
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

        # ── Cross-Modal Interaction Branch (2 Modalities) ─────────────────────
        self.cross_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # ── Fusion Regularisation ─────────────────────────────────────────────
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # ── Task Heads with Bottleneck MLP (Same as Exp 2) ────────────────────
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
        """
        Randomly drop modalities during training,
        ensuring at least one modality remains active per sample.
        """
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
        video_feat: torch.Tensor, # Passed but completely ignored!
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor, # Passed but completely ignored!
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


# ---------------------------------------------------------------------------
# Monkey-patches
# ---------------------------------------------------------------------------

# Patch 1: Test eval bypassing
original_evaluate = train_mod.evaluate
def patched_evaluate(model, dataloader, device, split, save_path=None):
    if split == "test":
        print("\n--- SKIPPING TEST EVALUATION AS REQUESTED ---")
        return {"split": "test", "skipped": True}
    return original_evaluate(model, dataloader, device, split, save_path)
train_mod.evaluate = patched_evaluate

import training.src.feature_cache as feature_cache
original_load_all_splits = feature_cache.load_all_splits
def patched_load_all_splits(modality, cache_dir):
    if modality == "video":
        print(">>> EXP 5A: Video modality intentionally suppressed (returning None) <<<")
        return None, None, None
    return original_load_all_splits(modality, cache_dir)
feature_cache.load_all_splits = patched_load_all_splits

# Patch 3: Use the new 2-modality model
def patched_build_model(device):
    model = GatedTextAudioFusion(
        text_dim=768, audio_dim=768,
        fusion_dim=512,
        emotion_classes=7, sentiment_classes=3,
        dropout=0.3,
        modality_dropout=0.05
    ).to(device)
    train_mod.count_parameters(model)
    return model
train_mod.build_model = patched_build_model

# ---------------------------------------------------------------------------
# Execute Sanity Run
# ---------------------------------------------------------------------------

print("=======================================================")
print("🚀 STARTING EXPERIMENT 5A SANITY RUN (3 EPOCHS)")
print("=======================================================")
print("Constraints applied:")
print("Experiment = 5A")
print("Active modalities:")
print("Text = YES")
print("Audio = YES")
print("Video = NO")
print(f"WeightedRandomSampler = NO")
print(f"Class-weighted loss = YES (POWER={CLASS_WEIGHT_POWER})")
print(f"Loss label smoothing = {LABEL_SMOOTHING}")
print("Emotion loss weight = 0.6")
print("Sentiment loss weight = 0.4")
print("Batch size = 64")
print("Learning rate = 7.5e-5")
print("CPU mode = YES")
print("Test evaluation = DISABLED")
print("=======================================================\n")

results = run_full_training(
    checkpoint_dir="data/exp5a_checkpoints",
    output_dir="data/exp5a_output",
    max_epochs=3,
    export_artifacts=False
)

print("\n=======================================================")
print("🏁 EXPERIMENT 5A SANITY RUN FINAL REPORT")
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
print(f"Exact Path of Best Checkpoint:   {os.path.abspath('data/exp5a_checkpoints/checkpoint_best.pt')}")
print("Test Set Bypassed:               Yes")
print("Run Completed Successfully:      Yes")
print("=======================================================\n")
