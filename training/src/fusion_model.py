"""
Affectra AI — Gated Multimodal Fusion Model
=============================================
Lightweight fusion network that combines text, audio, and video features
into a unified representation for emotion and sentiment classification.

Architecture:
  Text [768]  → Linear(768,256) + ReLU → gate → weighted [256]
  Audio [768] → Linear(768,256) + ReLU → gate → weighted [256]
  Video [768] → Linear(768,256) + ReLU → gate → weighted [256]
                ↓
          Sum of gated projections → [256]
          LayerNorm + Dropout
                ↓
   Emotion head: Linear(256,7)    → 7-class logits
   Sentiment head: Linear(256,3)  → 3-class logits

Key properties:
  - Only ~594K trainable parameters (encoders are frozen separately)
  - Handles missing modalities via binary masks (not None checks)
  - Gates are sigmoid outputs → soft attention per modality
  - Missing modalities contribute zero to the fused representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.src.config import (
    AUDIO_FEATURE_DIM,
    DROPOUT,
    FUSION_DIM,
    NUM_EMOTION_CLASSES,
    NUM_SENTIMENT_CLASSES,
    TEXT_FEATURE_DIM,
    VIDEO_FEATURE_DIM,
    get_model_config,
)
from training.src.utils import count_parameters, get_logger

logger = get_logger(__name__)


class ModalityProjection(nn.Module):
    """
    Residual projection block for a single modality feature.
    Maps 768-dim encoder embeddings into the shared fusion space (256-dim)
    with non-linear capacity, LayerNorm, and residual connection.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act(self.norm1(self.fc1(x)))
        h = self.drop(h)
        h = self.norm2(self.fc2(h) + res)
        return h * mask


class GatedMultimodalFusion(nn.Module):
    """
    Enhanced Context-Aware Gated Multimodal Fusion Network.

    Features:
      - Residual projections for each modality (prevents representation bottleneck)
      - Cross-modal contextual gating: gate scores are conditioned on both
        the modality's representation AND the joint multimodal context
      - Softmax-normalized gate weights to ensure stable fusion across any combination of modalities
      - Cross-modal interaction branch for non-linear inter-modality feature synthesis
      - Training-time modality dropout to prevent text dominance and overfitting
      - Bottleneck MLP task heads for emotion (7 classes) and sentiment (3 classes)
    """

    def __init__(
        self,
        input_dim: int = TEXT_FEATURE_DIM,
        fusion_dim: int = FUSION_DIM,
        num_emotions: int = NUM_EMOTION_CLASSES,
        num_sentiments: int = NUM_SENTIMENT_CLASSES,
        dropout: float = DROPOUT,
        modality_dropout: float = 0.15,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.fusion_dim = fusion_dim
        self.num_emotions = num_emotions
        self.num_sentiments = num_sentiments
        self.modality_dropout_prob = modality_dropout

        # ── Modality Projections with Residual MLP ───────────────────────────
        self.text_proj  = ModalityProjection(input_dim, fusion_dim, dropout)
        self.audio_proj = ModalityProjection(input_dim, fusion_dim, dropout)
        self.video_proj = ModalityProjection(input_dim, fusion_dim, dropout)

        # ── Cross-Modal Context Gates ─────────────────────────────────────────
        # Input to each gate is [modality_feature, global_context] (256 + 256 = 512)
        gate_in_dim = fusion_dim * 2
        self.text_gate = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )
        self.video_gate = nn.Sequential(
            nn.Linear(gate_in_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

        # ── Cross-Modal Interaction Branch ────────────────────────────────────
        self.cross_proj = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        # ── Fusion Regularisation ─────────────────────────────────────────────
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # ── Task Heads with Bottleneck MLP ────────────────────────────────────
        head_hidden = fusion_dim // 2
        self.emotion_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_emotions),
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_sentiments),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_modality_dropout(
        self,
        t_mask: torch.Tensor,
        a_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> tuple:
        """
        Randomly drop modalities during training with probability p,
        ensuring at least one modality remains active per sample.
        """
        if not self.training or self.modality_dropout_prob <= 0.0:
            return t_mask, a_mask, v_mask

        device = t_mask.device
        b = t_mask.size(0)

        # Random drop masks
        t_keep = (torch.rand(b, 1, device=device) > self.modality_dropout_prob).float()
        a_keep = (torch.rand(b, 1, device=device) > self.modality_dropout_prob).float()
        v_keep = (torch.rand(b, 1, device=device) > self.modality_dropout_prob).float()

        new_t = t_mask * t_keep
        new_a = a_mask * a_keep
        new_v = v_mask * v_keep

        # Guard: if all modalities were dropped for a sample, revert to original mask
        all_dropped = (new_t + new_a + new_v) < 0.5
        new_t = torch.where(all_dropped, t_mask, new_t)
        new_a = torch.where(all_dropped, a_mask, new_a)
        new_v = torch.where(all_dropped, v_mask, new_v)

        return new_t, new_a, new_v

    def forward(
        self,
        text_feat: torch.Tensor,    # [B, 768]
        audio_feat: torch.Tensor,   # [B, 768]
        video_feat: torch.Tensor,   # [B, 768]
        text_mask: torch.Tensor,    # [B] or [B, 1]
        audio_mask: torch.Tensor,   # [B] or [B, 1]
        video_mask: torch.Tensor,   # [B] or [B, 1]
    ) -> tuple:
        """
        Forward pass through the context-aware gated fusion network.
        """
        t_mask = text_mask.view(-1, 1).float()
        a_mask = audio_mask.view(-1, 1).float()
        v_mask = video_mask.view(-1, 1).float()

        # Apply modality dropout in training mode to prevent text over-reliance
        t_mask, a_mask, v_mask = self._apply_modality_dropout(t_mask, a_mask, v_mask)

        # ── 1. Residual Projections ──────────────────────────────────────────
        t_proj = self.text_proj(text_feat, t_mask)    # [B, 256]
        a_proj = self.audio_proj(audio_feat, a_mask)  # [B, 256]
        v_proj = self.video_proj(video_feat, v_mask)  # [B, 256]

        # ── 2. Joint Multimodal Context Summary ──────────────────────────────
        active_counts = (t_mask + a_mask + v_mask).clamp(min=1.0)
        context = (t_proj + a_proj + v_proj) / active_counts  # [B, 256]

        # ── 3. Cross-Modal Contextual Gates ──────────────────────────────────
        t_in = torch.cat([t_proj, context], dim=-1)
        a_in = torch.cat([a_proj, context], dim=-1)
        v_in = torch.cat([v_proj, context], dim=-1)

        t_gate = torch.sigmoid(self.text_gate(t_in)) * t_mask
        a_gate = torch.sigmoid(self.audio_gate(a_in)) * a_mask
        v_gate = torch.sigmoid(self.video_gate(v_in)) * v_mask

        # Normalize gate weights across active modalities
        gate_sum = (t_gate + a_gate + v_gate).clamp(min=1e-6)
        t_weight = t_gate / gate_sum
        a_weight = a_gate / gate_sum
        v_weight = v_gate / gate_sum

        gated_sum = t_weight * t_proj + a_weight * a_proj + v_weight * v_proj  # [B, 256]

        # ── 4. Cross-Modal Interaction Branch ────────────────────────────────
        all_modalities = torch.cat([t_proj, a_proj, v_proj], dim=-1)           # [B, 768]
        cross_inter = self.cross_proj(all_modalities)                          # [B, 256]

        # ── 5. Dual-Branch Fusion with Residual Link ─────────────────────────
        fused = gated_sum + cross_inter + context                              # [B, 256]
        fused = self.fusion_norm(self.fusion_dropout(fused))

        # ── 6. Task Heads ────────────────────────────────────────────────────
        emotion_logits = self.emotion_head(fused)        # [B, 7]
        sentiment_logits = self.sentiment_head(fused)    # [B, 3]

        return emotion_logits, sentiment_logits

    def get_gate_weights(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> dict:
        """
        Return the normalized gate weights for a batch.
        """
        with torch.no_grad():
            t_mask = text_mask.view(-1, 1).float()
            a_mask = audio_mask.view(-1, 1).float()
            v_mask = video_mask.view(-1, 1).float()

            t_proj = self.text_proj(text_feat, t_mask)
            a_proj = self.audio_proj(audio_feat, a_mask)
            v_proj = self.video_proj(video_feat, v_mask)

            active_counts = (t_mask + a_mask + v_mask).clamp(min=1.0)
            context = (t_proj + a_proj + v_proj) / active_counts

            t_gate = torch.sigmoid(self.text_gate(torch.cat([t_proj, context], dim=-1))) * t_mask
            a_gate = torch.sigmoid(self.audio_gate(torch.cat([a_proj, context], dim=-1))) * a_mask
            v_gate = torch.sigmoid(self.video_gate(torch.cat([v_proj, context], dim=-1))) * v_mask

            gate_sum = (t_gate + a_gate + v_gate).clamp(min=1e-6)

        return {
            "text_gate": t_gate / gate_sum,
            "audio_gate": a_gate / gate_sum,
            "video_gate": v_gate / gate_sum,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> GatedMultimodalFusion:
    """
    Build and return the fusion model, move to device, and print param count.
    """
    model = GatedMultimodalFusion().to(device)
    count_parameters(model)
    return model


# ---------------------------------------------------------------------------
# Balanced Multi-Task Weighted Loss
# ---------------------------------------------------------------------------

def build_weighted_loss(
    emotion_counts: dict,
    sentiment_counts: dict,
    device: torch.device,
    alpha: float = 0.6,
    beta: float = 0.4,
    power: float = 0.5,
    label_smoothing: float = 0.05,
):
    """
    Build smoothed class-weighted CrossEntropyLoss with label smoothing.

    Uses square-root frequency weighting:
      w_c = (total / (N_classes * count_c)) ** power
    Normalized so that mean(w) = 1.0.

    This prevents extreme loss spikes on rare classes (fear, disgust) while
    boosting their recall, avoiding the severe underfitting on rare classes
    caused by unweighted loss, and avoiding the optimization instability
    caused by raw linear inverse weighting.
    """
    from training.src.config import EMOTION_LABEL2ID, SENTIMENT_LABEL2ID

    def _compute_weights(counts: dict, label2id: dict) -> torch.Tensor:
        n_classes = len(label2id)
        weights = torch.ones(n_classes, dtype=torch.float32)
        total = sum(counts.values())
        for label, idx in label2id.items():
            count = counts.get(label, 1)
            raw_w = total / (n_classes * max(count, 1))
            weights[idx] = raw_w ** power
        weights = weights / weights.mean()
        return weights.to(device)

    emotion_weights   = _compute_weights(emotion_counts, EMOTION_LABEL2ID)
    sentiment_weights = _compute_weights(sentiment_counts, SENTIMENT_LABEL2ID)

    emotion_criterion = nn.CrossEntropyLoss(
        weight=emotion_weights,
        label_smoothing=label_smoothing,
    )
    sentiment_criterion = nn.CrossEntropyLoss(
        weight=sentiment_weights,
        label_smoothing=label_smoothing,
    )

    logger.info(f"Emotion class weights:   {[round(w, 3) for w in emotion_weights.cpu().tolist()]}")
    logger.info(f"Sentiment class weights: {[round(w, 3) for w in sentiment_weights.cpu().tolist()]}")

    return emotion_criterion, sentiment_criterion, alpha, beta


if __name__ == "__main__":
    # Quick smoke test
    device = torch.device("cpu")
    model = build_model(device)

    B = 4
    text_feat  = torch.randn(B, 768)
    audio_feat = torch.randn(B, 768)
    video_feat = torch.randn(B, 768)

    # Test: all modalities present
    t_mask = torch.ones(B)
    a_mask = torch.ones(B)
    v_mask = torch.ones(B)

    e_logits, s_logits = model(text_feat, audio_feat, video_feat, t_mask, a_mask, v_mask)
    print(f"Emotion logits shape:   {e_logits.shape}  (expected [4, 7])")
    print(f"Sentiment logits shape: {s_logits.shape}  (expected [4, 3])")

    # Test: text only (audio + video masks = 0)
    a_mask_off = torch.zeros(B)
    v_mask_off = torch.zeros(B)
    e2, s2 = model(text_feat, audio_feat, video_feat, t_mask, a_mask_off, v_mask_off)
    print(f"Text-only forward:      ✅  shapes {e2.shape}, {s2.shape}")

    print("✅ GatedMultimodalFusion smoke test passed.")
