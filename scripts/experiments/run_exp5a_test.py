import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from training.src.config import EMOTION_LABELS, SENTIMENT_LABELS
import training.src.train as train_mod
from training.src.dataset import load_meld_metadata
from training.src.feature_cache import load_all_splits

from training.src.fusion_model import ModalityProjection

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

        self.text_proj = ModalityProjection(text_dim, fusion_dim, dropout)
        self.audio_proj = ModalityProjection(audio_dim, fusion_dim, dropout)

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

        self.cross_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout)

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

print("==================================================")
print("1. LOAD EXPERIMENT 5A BEST CHECKPOINT")
print("==================================================")

device = torch.device("cpu")
model = GatedTextAudioFusion(
    text_dim=768, audio_dim=768,
    fusion_dim=512,
    emotion_classes=7, sentiment_classes=3,
    dropout=0.3,
    modality_dropout=0.05
).to(device)

ckpt_path = "data/exp5a_checkpoints/checkpoint_best.pt"
train_mod.load_checkpoint(ckpt_path, model, device=device)
model.eval()

print("\n==================================================")
print("2. LOAD TEST SET ONLY (TEXT + AUDIO)")
print("==================================================")

out_dir = "data/exp5a_output"
os.makedirs(out_dir, exist_ok=True)

# Load cached test features (ONLY Text + Audio)
_, _, test_text = load_all_splits("text", "data/feature_cache")
_, _, test_audio = load_all_splits("audio", "data/feature_cache")

test_df = load_meld_metadata("data/MELD/annotations/test_sent_emo.csv")

assert len(test_df) == 2610, f"Expected 2610 test samples, got {len(test_df)}"
assert test_text.shape == (2610, 768)
assert test_audio.shape == (2610, 768)

print("✅ Test size = 2610")
print("✅ Text test shape = [2610, 768]")
print("✅ Audio test shape = [2610, 768]")
print("✅ Video intentionally ignored")

# Generate predictions
e_labels = test_df["emotion_id"].values
s_labels = test_df["sentiment_id"].values
e_preds = []
s_preds = []

with torch.no_grad():
    for i in range(0, 2610, 64):
        t_feat = test_text[i:i+64].to(device)
        a_feat = test_audio[i:i+64].to(device)
        v_feat = torch.zeros_like(t_feat) # Dummy video feat (ignored by architecture)
        t_mask = torch.ones(t_feat.size(0)).to(device)
        a_mask = torch.ones(t_feat.size(0)).to(device)
        v_mask = torch.zeros(t_feat.size(0)).to(device) # Dummy video mask
        
        e_logits, s_logits = model(t_feat, a_feat, v_feat, t_mask, a_mask, v_mask)
        
        e_preds.append(torch.argmax(e_logits, dim=1).cpu().numpy())
        s_preds.append(torch.argmax(s_logits, dim=1).cpu().numpy())

e_preds = np.concatenate(e_preds)
s_preds = np.concatenate(s_preds)

print("\n==================================================")
print("3. FINAL METRICS")
print("==================================================")

emo_acc = accuracy_score(e_labels, e_preds)
emo_p, emo_r, emo_f, _ = precision_recall_fscore_support(e_labels, e_preds, average='weighted', zero_division=0)
emo_p_m, emo_r_m, emo_f_m, _ = precision_recall_fscore_support(e_labels, e_preds, average='macro', zero_division=0)

sent_acc = accuracy_score(s_labels, s_preds)
sent_p, sent_r, sent_f, _ = precision_recall_fscore_support(s_labels, s_preds, average='weighted', zero_division=0)
sent_p_m, sent_r_m, sent_f_m, _ = precision_recall_fscore_support(s_labels, s_preds, average='macro', zero_division=0)

report_emo = classification_report(e_labels, e_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)

metrics_json = {
    "EMOTION": {
        "accuracy": emo_acc,
        "weighted_precision": emo_p,
        "weighted_recall": emo_r,
        "weighted_f1": emo_f,
        "macro_precision": emo_p_m,
        "macro_recall": emo_r_m,
        "macro_f1": emo_f_m
    },
    "SENTIMENT": {
        "accuracy": sent_acc,
        "weighted_precision": sent_p,
        "weighted_recall": sent_r,
        "weighted_f1": sent_f,
        "macro_precision": sent_p_m,
        "macro_recall": sent_r_m,
        "macro_f1": sent_f_m
    },
    "PER_CLASS_EMOTION": {
        label: report_emo[label] for label in EMOTION_LABELS
    }
}

with open(os.path.join(out_dir, "final_test_metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=4)

print("\n==================================================")
print("4. CONFUSION MATRIX")
print("==================================================")

cm = confusion_matrix(e_labels, e_preds, labels=range(7))
cm_norm = confusion_matrix(e_labels, e_preds, labels=range(7), normalize='true')

cm_df = pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS)
cm_norm_df = pd.DataFrame(cm_norm, index=EMOTION_LABELS, columns=EMOTION_LABELS)

cm_df.to_csv(os.path.join(out_dir, "final_test_emotion_confusion_matrix.csv"))
cm_norm_df.to_csv(os.path.join(out_dir, "final_test_emotion_confusion_matrix_normalized.csv"))

print("\n==================================================")
print("5. OUTPUT")
print("==================================================")

md_content = f"""# Affectra-AI Experiment 5A - Final Test Report

## 1. Overall Test Metrics

### EMOTION
- Accuracy: {emo_acc*100:.2f}%
- Weighted Precision: {emo_p*100:.2f}%
- Weighted Recall: {emo_r*100:.2f}%
- Weighted F1: {emo_f*100:.2f}%
- Macro Precision: {emo_p_m*100:.2f}%
- Macro Recall: {emo_r_m*100:.2f}%
- Macro F1: {emo_f_m*100:.2f}%

### SENTIMENT
- Accuracy: {sent_acc*100:.2f}%
- Weighted Precision: {sent_p*100:.2f}%
- Weighted Recall: {sent_r*100:.2f}%
- Weighted F1: {sent_f*100:.2f}%
- Macro Precision: {sent_p_m*100:.2f}%
- Macro Recall: {sent_r_m*100:.2f}%
- Macro F1: {sent_f_m*100:.2f}%

## 2. Per-Class Emotion Results
"""

for label in EMOTION_LABELS:
    c = report_emo[label]
    md_content += f"""
### {label}
- Precision: {c['precision']*100:.2f}%
- Recall: {c['recall']*100:.2f}%
- F1: {c['f1-score']*100:.2f}%
- Support: {c['support']}
"""

md_content += f"""
## 3. Comparison with Experiment 2
| Metric | Exp 2 (Test) | Exp 5A (Test) | Difference |
| :--- | :--- | :--- | :--- |
| Emotion Accuracy | 49.77% | {emo_acc*100:.2f}% | {(emo_acc - 0.4977)*100:+.2f}% |
| Emotion wF1 | 52.71% | {emo_f*100:.2f}% | {(emo_f - 0.5271)*100:+.2f}% |
| Emotion mF1 | 37.28% | {emo_f_m*100:.2f}% | {(emo_f_m - 0.3728)*100:+.2f}% |
| Sentiment Accuracy | 65.56% | {sent_acc*100:.2f}% | {(sent_acc - 0.6556)*100:+.2f}% |
| Sentiment wF1 | 65.66% | {sent_f*100:.2f}% | {(sent_f - 0.6566)*100:+.2f}% |
| Sentiment mF1 | 62.91% | {sent_f_m*100:.2f}% | {(sent_f_m - 0.6291)*100:+.2f}% |

## 4. Generalization
Exp 5A Dev Emotion wF1: 54.51%
Exp 5A Test Emotion wF1: {emo_f*100:.2f}%
"""

with open(os.path.join(out_dir, "FINAL_TEST_REPORT.md"), "w") as f:
    f.write(md_content)

print("1. Test emotion accuracy: {:.2f}%".format(emo_acc*100))
print("2. Test emotion weighted F1: {:.2f}%".format(emo_f*100))
print("3. Test emotion macro F1: {:.2f}%".format(emo_f_m*100))
print("4. Test sentiment accuracy: {:.2f}%".format(sent_acc*100))
print("5. Test sentiment weighted F1: {:.2f}%".format(sent_f*100))
print("6. Test sentiment macro F1: {:.2f}%".format(sent_f_m*100))

print("\n7. Per-emotion F1:")
for label in EMOTION_LABELS:
    print(f"   {label}: {report_emo[label]['f1-score']*100:.2f}%")

print("\n8. Comparison against Experiment 2 TEST:")
print(f"   Emotion Accuracy: 49.77% -> {emo_acc*100:.2f}% ({(emo_acc - 0.4977)*100:+.2f}%)")
print(f"   Emotion wF1:      52.71% -> {emo_f*100:.2f}% ({(emo_f - 0.5271)*100:+.2f}%)")
print(f"   Emotion mF1:      37.28% -> {emo_f_m*100:.2f}% ({(emo_f_m - 0.3728)*100:+.2f}%)")
print(f"   Sentiment Accuracy: 65.56% -> {sent_acc*100:.2f}% ({(sent_acc - 0.6556)*100:+.2f}%)")
print(f"   Sentiment wF1:      65.66% -> {sent_f*100:.2f}% ({(sent_f - 0.6566)*100:+.2f}%)")
print(f"   Sentiment mF1:      62.91% -> {sent_f_m*100:.2f}% ({(sent_f_m - 0.6291)*100:+.2f}%)")

gen_gap = (0.5451 - emo_f) * 100
gen_status = "YES" if gen_gap < 3.0 else "NO (Overfitting)"
print(f"\n9. Whether Experiment 5A generalizes from DEV to TEST: {gen_status} (Dev=54.51%, Test={emo_f*100:.2f}%, Gap={gen_gap:+.2f}%)")
