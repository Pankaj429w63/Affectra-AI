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


class GatedMultimodalFusion(nn.Module):
    """
    Gated multimodal fusion network for emotion and sentiment classification.

    Args:
        input_dim:      Feature dimension output by each encoder (default 768).
        fusion_dim:     Dimension of the projected fusion space (default 256).
        num_emotions:   Number of emotion classes (default 7).
        num_sentiments: Number of sentiment classes (default 3).
        dropout:        Dropout probability in fusion layer (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = TEXT_FEATURE_DIM,
        fusion_dim: int = FUSION_DIM,
        num_emotions: int = NUM_EMOTION_CLASSES,
        num_sentiments: int = NUM_SENTIMENT_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.fusion_dim = fusion_dim
        self.num_emotions = num_emotions
        self.num_sentiments = num_sentiments

        # ── Modality Projections ────────────────────────────────────────────
        # Each modality's 768-dim feature is projected to fusion_dim (256)
        self.text_proj  = nn.Linear(input_dim, fusion_dim)
        self.audio_proj = nn.Linear(input_dim, fusion_dim)
        self.video_proj = nn.Linear(input_dim, fusion_dim)

        # ── Learned Scalar Gates ─────────────────────────────────────────────
        # Each gate produces a [B, 1] scalar weight via sigmoid activation.
        # The model learns which modalities are most informative.
        self.text_gate  = nn.Linear(fusion_dim, 1)
        self.audio_gate = nn.Linear(fusion_dim, 1)
        self.video_gate = nn.Linear(fusion_dim, 1)

        # ── Fusion Layer ─────────────────────────────────────────────────────
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(fusion_dim)

        # ── Task Heads ────────────────────────────────────────────────────────
        self.emotion_head    = nn.Linear(fusion_dim, num_emotions)
        self.sentiment_head  = nn.Linear(fusion_dim, num_sentiments)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_feat: torch.Tensor,    # [B, 768]
        audio_feat: torch.Tensor,   # [B, 768]
        video_feat: torch.Tensor,   # [B, 768]
        text_mask: torch.Tensor,    # [B] or [B, 1]  — 1.0 if available
        audio_mask: torch.Tensor,   # [B] or [B, 1]
        video_mask: torch.Tensor,   # [B] or [B, 1]
    ) -> tuple:
        """
        Forward pass through the gated fusion network.

        Modality masks (0/1 floats) zero out the contribution of
        unavailable modalities before gating. This means the gates
        for missing modalities output 0 regardless of learned weights.

        Args:
            text_feat:    Text encoder output [B, 768].
            audio_feat:   Audio encoder output [B, 768].
            video_feat:   Video encoder output [B, 768].
            text_mask:    Binary mask [B] — 1.0 if text is available.
            audio_mask:   Binary mask [B] — 1.0 if audio is available.
            video_mask:   Binary mask [B] — 1.0 if video is available.

        Returns:
            Tuple[Tensor, Tensor]:
                emotion_logits    [B, 7]  — raw scores for 7 emotion classes
                sentiment_logits  [B, 3]  — raw scores for 3 sentiment classes
        """
        # Ensure masks are [B, 1] for broadcasting
        t_mask = text_mask.view(-1, 1).float()   # [B, 1]
        a_mask = audio_mask.view(-1, 1).float()  # [B, 1]
        v_mask = video_mask.view(-1, 1).float()  # [B, 1]

        # ── Project each modality ─────────────────────────────────────────
        t_proj = self.activation(self.text_proj(text_feat))    # [B, 256]
        a_proj = self.activation(self.audio_proj(audio_feat))  # [B, 256]
        v_proj = self.activation(self.video_proj(video_feat))  # [B, 256]

        # ── Compute gates ─────────────────────────────────────────────────
        # Multiply by mask BEFORE gate so missing modalities gate stays 0
        t_gate = torch.sigmoid(self.text_gate(t_proj * t_mask))   # [B, 1]
        a_gate = torch.sigmoid(self.audio_gate(a_proj * a_mask))  # [B, 1]
        v_gate = torch.sigmoid(self.video_gate(v_proj * v_mask))  # [B, 1]

        # Apply mask to gate output (belt-and-suspenders)
        t_gate = t_gate * t_mask
        a_gate = a_gate * a_mask
        v_gate = v_gate * v_mask

        # ── Gated weighted sum ────────────────────────────────────────────
        fused = (
            t_gate * t_proj +
            a_gate * a_proj +
            v_gate * v_proj
        )  # [B, 256]

        # ── Normalise and regularise ──────────────────────────────────────
        fused = self.layer_norm(self.dropout(fused))  # [B, 256]

        # ── Task heads ────────────────────────────────────────────────────
        emotion_logits   = self.emotion_head(fused)    # [B, 7]
        sentiment_logits = self.sentiment_head(fused)  # [B, 3]

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
        Return the gate weights for a batch (useful for debugging and
        understanding which modalities the model relies on most).

        Returns:
            dict with keys 'text_gate', 'audio_gate', 'video_gate'
            — each a FloatTensor [B, 1].
        """
        with torch.no_grad():
            t_mask = text_mask.view(-1, 1).float()
            a_mask = audio_mask.view(-1, 1).float()
            v_mask = video_mask.view(-1, 1).float()

            t_proj = self.activation(self.text_proj(text_feat))
            a_proj = self.activation(self.audio_proj(audio_feat))
            v_proj = self.activation(self.video_proj(video_feat))

            t_gate = torch.sigmoid(self.text_gate(t_proj * t_mask)) * t_mask
            a_gate = torch.sigmoid(self.audio_gate(a_proj * a_mask)) * a_mask
            v_gate = torch.sigmoid(self.video_gate(v_proj * v_mask)) * v_mask

        return {
            "text_gate": t_gate,
            "audio_gate": a_gate,
            "video_gate": v_gate,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> GatedMultimodalFusion:
    """
    Build and return the fusion model, move to device, and print param count.

    Args:
        device: torch.device to place the model on.

    Returns:
        GatedMultimodalFusion ready for training.
    """
    model = GatedMultimodalFusion().to(device)
    count_parameters(model)
    return model


# ---------------------------------------------------------------------------
# Weighted Loss
# ---------------------------------------------------------------------------

def build_weighted_loss(
    emotion_counts: dict,
    sentiment_counts: dict,
    device: torch.device,
    alpha: float = 0.6,
    beta: float = 0.4,
):
    """
    Build class-weighted CrossEntropyLoss for both task heads.

    Weights are the inverse of class frequency, normalised so the mean = 1.
    This gives under-represented classes (e.g., 'fear', 'disgust') higher
    weight in the loss, compensating for MELD's class imbalance.

    Args:
        emotion_counts:   Dict mapping emotion label string → count.
        sentiment_counts: Dict mapping sentiment label string → count.
        device:           Device to place weight tensors on.
        alpha:            Weight for the emotion loss term.
        beta:             Weight for the sentiment loss term.

    Returns:
        Tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss, float, float]:
          (emotion_criterion, sentiment_criterion, alpha, beta)
    """
    from training.src.config import EMOTION_LABEL2ID, SENTIMENT_LABEL2ID

    def _compute_weights(counts: dict, label2id: dict) -> torch.Tensor:
        n_classes = len(label2id)
        weights = torch.ones(n_classes, dtype=torch.float32)
        total = sum(counts.values())
        for label, idx in label2id.items():
            count = counts.get(label, 1)
            # Inverse frequency weight
            weights[idx] = total / (n_classes * max(count, 1))
        # Normalise so mean weight = 1 (keeps loss scale stable)
        weights = weights / weights.mean()
        return weights.to(device)

    emotion_weights   = _compute_weights(emotion_counts, EMOTION_LABEL2ID)
    sentiment_weights = _compute_weights(sentiment_counts, SENTIMENT_LABEL2ID)

    emotion_criterion   = nn.CrossEntropyLoss(weight=emotion_weights)
    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)

    logger.info(f"Emotion class weights:   {emotion_weights.cpu().tolist()}")
    logger.info(f"Sentiment class weights: {sentiment_weights.cpu().tolist()}")

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
