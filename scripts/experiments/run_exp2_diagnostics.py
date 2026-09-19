import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from training.src.config import EMOTION_LABELS, SENTIMENT_LABELS
from training.src.fusion_model import build_model
import training.src.train as train_mod
from training.src.dataset import MELDCachedDataset
from torch.utils.data import DataLoader

print("==================================================")
print("1. LOAD EXPERIMENT 2 BEST CHECKPOINT")
print("==================================================")

device = torch.device("cpu")
model = build_model(device=device)
ckpt_path = "data/exp2_checkpoints/checkpoint_best.pt"
train_mod.load_checkpoint(ckpt_path, model, device=device)
model.eval()

# Verify parameter count
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {param_count:,}")
assert param_count == 5324813, "Parameter count mismatch!"

print("\n==================================================")
print("2. USE DEV SET ONLY")
print("==================================================")

out_dir = "data/exp2_diagnostics"
os.makedirs(out_dir, exist_ok=True)

# Load cached dev features
from training.src.feature_cache import load_all_splits
train_text, dev_text, test_text = load_all_splits("text", "data/feature_cache")
train_audio, dev_audio, test_audio = load_all_splits("audio", "data/feature_cache")
train_video, dev_video, test_video = load_all_splits("video", "data/feature_cache")

from training.src.dataset import load_meld_metadata
dev_df = load_meld_metadata("data/MELD/annotations/dev_sent_emo.csv")

assert len(dev_df) == 1109, f"Expected 1109 dev samples, got {len(dev_df)}"
assert dev_text.shape == (1109, 768)
assert dev_audio.shape == (1109, 768)
assert dev_video.shape == (1109, 768)

dev_ds = MELDCachedDataset(dev_df, dev_text, dev_audio, dev_video)
dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False)

def run_inference(ablate_text=False, ablate_audio=False, ablate_video=False):
    all_emo_preds = []
    all_emo_probs = []
    all_emo_labels = []
    all_sent_preds = []
    all_sent_labels = []
    
    with torch.no_grad():
        for batch in dev_loader:
            t_feat = batch["text_feat"].to(device)
            a_feat = batch["audio_feat"].to(device)
            v_feat = batch["video_feat"].to(device)
            t_mask = batch["text_mask"].to(device)
            a_mask = batch["audio_mask"].to(device)
            v_mask = batch["video_mask"].to(device)
            
            if ablate_text: t_mask.zero_()
            if ablate_audio: a_mask.zero_()
            if ablate_video: v_mask.zero_()
                
            e_logits, s_logits = model(t_feat, a_feat, v_feat, t_mask, a_mask, v_mask)
            
            e_probs = F.softmax(e_logits, dim=1)
            e_preds = torch.argmax(e_logits, dim=1)
            s_preds = torch.argmax(s_logits, dim=1)
            
            all_emo_probs.append(e_probs.cpu().numpy())
            all_emo_preds.append(e_preds.cpu().numpy())
            all_emo_labels.append(batch["emotion_label"].cpu().numpy())
            all_sent_preds.append(s_preds.cpu().numpy())
            all_sent_labels.append(batch["sentiment_label"].cpu().numpy())
            
    return {
        "emo_probs": np.concatenate(all_emo_probs),
        "emo_preds": np.concatenate(all_emo_preds),
        "emo_labels": np.concatenate(all_emo_labels),
        "sent_preds": np.concatenate(all_sent_preds),
        "sent_labels": np.concatenate(all_sent_labels),
    }

# Baseline inference
res_full = run_inference()
e_labels = res_full["emo_labels"]
e_preds = res_full["emo_preds"]
e_probs = res_full["emo_probs"]

print("\n==================================================")
print("3. CONFUSION MATRIX — EMOTION")
print("==================================================")

cm = confusion_matrix(e_labels, e_preds, labels=range(7))
cm_norm = confusion_matrix(e_labels, e_preds, labels=range(7), normalize='true')

cm_df = pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS)
cm_norm_df = pd.DataFrame(cm_norm, index=EMOTION_LABELS, columns=EMOTION_LABELS)

cm_df.to_csv(os.path.join(out_dir, "emotion_confusion_matrix.csv"))
cm_norm_df.to_csv(os.path.join(out_dir, "emotion_confusion_matrix_normalized.csv"))

print("Raw Confusion Matrix:")
print(cm_df)

print("\n==================================================")
print("4. PER-CLASS EMOTION METRICS")
print("==================================================")

report = classification_report(e_labels, e_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(out_dir, "emotion_per_class_metrics.csv"))

print(f"Dev Emotion Weighted F1: {report['weighted avg']['f1-score']*100:.2f}%")
print(f"Dev Emotion Macro F1:    {report['macro avg']['f1-score']*100:.2f}%")
print(f"Dev Emotion Accuracy:    {accuracy_score(e_labels, e_preds)*100:.2f}%")

print("\n==================================================")
print("5. PREDICTION DISTRIBUTION")
print("==================================================")

actual_counts = Counter(e_labels)
pred_counts = Counter(e_preds)

dist_data = []
for i, label in enumerate(EMOTION_LABELS):
    ac = actual_counts.get(i, 0)
    pc = pred_counts.get(i, 0)
    dist_data.append({
        "emotion": label,
        "actual_count": ac,
        "actual_percentage": ac / len(e_labels) * 100,
        "predicted_count": pc,
        "predicted_percentage": pc / len(e_preds) * 100,
        "prediction_minus_actual": pc - ac
    })

dist_df = pd.DataFrame(dist_data)
dist_df.to_csv(os.path.join(out_dir, "emotion_prediction_distribution.csv"), index=False)

print("\n==================================================")
print("6. ERROR ANALYSIS FOR DISGUST AND FEAR")
print("==================================================")

def extract_errors(target_idx):
    target_name = EMOTION_LABELS[target_idx]
    errors = []
    
    for i in range(len(e_labels)):
        if e_labels[i] == target_idx and e_preds[i] != target_idx:
            row = dev_df.iloc[i]
            errors.append({
                "true_emotion": target_name,
                "predicted_emotion": EMOTION_LABELS[e_preds[i]],
                "predicted_confidence": e_probs[i][e_preds[i]],
                "true_class_probability": e_probs[i][target_idx],
                "utterance": row["Utterance"],
                "speaker": row["Speaker"],
                "dialogue_id": row["Dialogue_ID"],
                "utterance_id": row["Utterance_ID"]
            })
    return pd.DataFrame(errors)

disgust_idx = EMOTION_LABELS.index("disgust")
fear_idx = EMOTION_LABELS.index("fear")

disgust_errors = extract_errors(disgust_idx)
fear_errors = extract_errors(fear_idx)

disgust_errors.to_csv(os.path.join(out_dir, "disgust_errors.csv"), index=False)
fear_errors.to_csv(os.path.join(out_dir, "fear_errors.csv"), index=False)

print(f"Disgust errors: {len(disgust_errors)}")
if len(disgust_errors) > 0:
    print("Most common disgust misclassifications:")
    print(disgust_errors["predicted_emotion"].value_counts())

print(f"\nFear errors: {len(fear_errors)}")
if len(fear_errors) > 0:
    print("Most common fear misclassifications:")
    print(fear_errors["predicted_emotion"].value_counts())

print("\n==================================================")
print("7. TOP CONFUSIONS")
print("==================================================")

confusions = []
for i in range(7):
    for j in range(7):
        if i != j and cm[i, j] > 0:
            confusions.append({
                "true_class": EMOTION_LABELS[i],
                "predicted_class": EMOTION_LABELS[j],
                "error_count": cm[i, j],
                "percentage_of_true_class": cm[i, j] / max(cm[i, :].sum(), 1) * 100
            })

conf_df = pd.DataFrame(confusions)
conf_df = conf_df.sort_values(by=["error_count", "percentage_of_true_class"], ascending=[False, False]).head(15)
conf_df.to_csv(os.path.join(out_dir, "top_emotion_confusions.csv"), index=False)

print("\n==================================================")
print("8. CONFIDENCE ANALYSIS")
print("==================================================")

conf_data = []
for i, label in enumerate(EMOTION_LABELS):
    idx_mask = (e_labels == i)
    if not np.any(idx_mask): continue
    
    probs = e_probs[idx_mask]
    preds = e_preds[idx_mask]
    correct_mask = (preds == i)
    
    avg_conf = np.mean(np.max(probs, axis=1))
    avg_correct_conf = np.mean(np.max(probs[correct_mask], axis=1)) if np.any(correct_mask) else 0.0
    avg_incorrect_conf = np.mean(np.max(probs[~correct_mask], axis=1)) if np.any(~correct_mask) else 0.0
    
    conf_data.append({
        "emotion": label,
        "avg_predicted_confidence": avg_conf,
        "avg_confidence_correct": avg_correct_conf,
        "avg_confidence_incorrect": avg_incorrect_conf
    })

conf_analysis_df = pd.DataFrame(conf_data)
conf_analysis_df.to_csv(os.path.join(out_dir, "emotion_confidence_analysis.csv"), index=False)

print("\n==================================================")
print("9. MODALITY ABLATION DIAGNOSTIC")
print("==================================================")

conditions = {
    "A. Text + Audio + Video": (False, False, False),
    "B. Text only": (False, True, True),
    "C. Audio only": (True, False, True),
    "D. Video only": (True, True, False),
    "E. Text + Audio": (False, False, True),
    "F. Text + Video": (False, True, False),
    "G. Audio + Video": (True, False, False),
}

ablation_metrics = []
rare_class_metrics = []

for cond_name, ablate_flags in conditions.items():
    res = run_inference(*ablate_flags)
    
    e_acc = accuracy_score(res["emo_labels"], res["emo_preds"])
    e_wf1 = f1_score(res["emo_labels"], res["emo_preds"], average="weighted", zero_division=0)
    e_mf1 = f1_score(res["emo_labels"], res["emo_preds"], average="macro", zero_division=0)
    
    s_acc = accuracy_score(res["sent_labels"], res["sent_preds"])
    s_wf1 = f1_score(res["sent_labels"], res["sent_preds"], average="weighted", zero_division=0)
    s_mf1 = f1_score(res["sent_labels"], res["sent_preds"], average="macro", zero_division=0)
    
    ablation_metrics.append({
        "Condition": cond_name,
        "Emotion Accuracy": e_acc,
        "Emotion Weighted F1": e_wf1,
        "Emotion Macro F1": e_mf1,
        "Sentiment Accuracy": s_acc,
        "Sentiment Weighted F1": s_wf1,
        "Sentiment Macro F1": s_mf1,
    })
    
    if cond_name in ["A. Text + Audio + Video", "B. Text only", "C. Audio only", "D. Video only"]:
        rep = classification_report(res["emo_labels"], res["emo_preds"], target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
        rare_class_metrics.append({
            "Condition": cond_name,
            "Disgust F1": rep["disgust"]["f1-score"],
            "Disgust Precision": rep["disgust"]["precision"],
            "Disgust Recall": rep["disgust"]["recall"],
            "Fear F1": rep["fear"]["f1-score"],
            "Fear Precision": rep["fear"]["precision"],
            "Fear Recall": rep["fear"]["recall"],
        })

ablation_df = pd.DataFrame(ablation_metrics)
ablation_df.to_csv(os.path.join(out_dir, "modality_ablation_metrics.csv"), index=False)

rare_df = pd.DataFrame(rare_class_metrics)
rare_df.to_csv(os.path.join(out_dir, "rare_class_modality_analysis.csv"), index=False)

print("\n==================================================")
print("12. FINAL DIAGNOSTIC REPORT (Markdown Generation)")
print("==================================================")

# Generate markdown report
md_content = f"""# Affectra-AI Experiment 2 Diagnostic Report

## 1. Executive Summary
This report analyzes the limitations of the Experiment 2 checkpoint on the dev set, specifically focusing on the failure modes for minority classes (disgust, fear).

## 2. Overall Emotion Metrics
- **Accuracy:** {report['accuracy']*100:.2f}%
- **Weighted F1:** {report['weighted avg']['f1-score']*100:.2f}%
- **Macro F1:** {report['macro avg']['f1-score']*100:.2f}%

## 3. Per-class Metrics
{report_df.to_markdown()}

## 4. Confusion Matrix Findings
{cm_df.to_markdown()}

## 5. Prediction Distribution
{dist_df.to_markdown()}

## 6. Top Confusion Pairs
{conf_df.to_markdown()}

## 7. Confidence Analysis
{conf_analysis_df.to_markdown()}

## 8. Modality Ablation
{ablation_df.to_markdown()}

## 9. Rare-Class Modality Analysis
{rare_df.to_markdown()}
"""

with open(os.path.join(out_dir, "EXP2_DIAGNOSTIC_REPORT.md"), "w") as f:
    f.write(md_content)

print("\n==================================================")
print("FINAL TERMINAL OUTPUT")
print("==================================================")

# Calculate some terminal highlights
best_emo = report_df[:-3].sort_values(by="f1-score", ascending=False).index[0]
worst_emo = report_df[:-3].sort_values(by="f1-score", ascending=True).index[0]

text_only_row = ablation_df[ablation_df["Condition"] == "B. Text only"].iloc[0]
audio_only_row = ablation_df[ablation_df["Condition"] == "C. Audio only"].iloc[0]
video_only_row = ablation_df[ablation_df["Condition"] == "D. Video only"].iloc[0]
best_two = ablation_df.iloc[4:7].sort_values(by="Emotion Weighted F1", ascending=False).iloc[0]

print(f"- Dev emotion accuracy: {report['accuracy']*100:.2f}%")
print(f"- Dev emotion weighted F1: {report['weighted avg']['f1-score']*100:.2f}%")
print(f"- Dev emotion macro F1: {report['macro avg']['f1-score']*100:.2f}%")
print(f"- Best/worst emotion classes: {best_emo} (best) / {worst_emo} (worst)")
print(f"- Disgust F1: {report['disgust']['f1-score']*100:.2f}%")
print(f"- Fear F1: {report['fear']['f1-score']*100:.2f}%")

print("\n- Top 5 confusion pairs:")
for i, row in conf_df.head(5).iterrows():
    print(f"  {row['true_class']} -> {row['predicted_class']} ({row['error_count']} errors)")

print("\n- Predicted-vs-actual distribution highlights:")
for i, row in dist_df.iterrows():
    if abs(row["prediction_minus_actual"]) > 20:
        over_under = "OVER-predicting" if row["prediction_minus_actual"] > 0 else "UNDER-predicting"
        print(f"  {row['emotion']}: {over_under} by {abs(row['prediction_minus_actual'])} samples")

print(f"\n- Text-only emotion F1: {text_only_row['Emotion Weighted F1']*100:.2f}%")
print(f"- Audio-only emotion F1: {audio_only_row['Emotion Weighted F1']*100:.2f}%")
print(f"- Video-only emotion F1: {video_only_row['Emotion Weighted F1']*100:.2f}%")
print(f"- Best two-modality combination: {best_two['Condition']} ({best_two['Emotion Weighted F1']*100:.2f}%)")
print(f"- Full multimodal emotion F1: {report['weighted avg']['f1-score']*100:.2f}%")

neutral_overpred = dist_df[dist_df["emotion"] == "neutral"]["prediction_minus_actual"].iloc[0]
biased = "YES" if neutral_overpred > 50 else "NO"
print(f"- Whether the model appears majority-class biased: {biased}")

audio_helps_fear = rare_df[rare_df["Condition"] == "A. Text + Audio + Video"]["Fear F1"].iloc[0] > rare_df[rare_df["Condition"] == "B. Text only"]["Fear F1"].iloc[0]
video_helps_disgust = rare_df[rare_df["Condition"] == "A. Text + Audio + Video"]["Disgust F1"].iloc[0] > rare_df[rare_df["Condition"] == "B. Text only"]["Disgust F1"].iloc[0]
print(f"- Whether audio/video appear helpful for disgust/fear: Audio helps fear: {audio_helps_fear}, Video helps disgust: {video_helps_disgust}")

print("\n- Recommended direction for Experiment 5:")
if text_only_row['Emotion Weighted F1'] > report['weighted avg']['f1-score']:
    print("  The textual modality dominates and fusion hurts. Focus on better text features or early fusion.")
elif biased == "YES":
    print("  The model strongly over-predicts neutral/joy. Since explicit class weighting/sampling failed, consider Focal Loss to target hard examples dynamically, or data augmentation.")
else:
    print("  Consider adjusting the contextual gating or using Focal Loss to better separate rare classes without destabilizing gradients.")
