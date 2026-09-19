import os
import sys
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.abspath("."))
from training.src.inference_model import AffectraPredictor
from training.src.feature_cache import load_all_splits
from training.src.dataset import load_meld_metadata

def main():
    print("==================================================")
    print("1. LOADING EXPORTED MODEL")
    print("==================================================")
    
    predictor = AffectraPredictor("models/affectra_multimodal", device="cpu")
    print("Model loaded successfully.")
    
    print("\n==================================================")
    print("2. INFERENCE SMOKE TEST (First 4 samples)")
    print("==================================================")
    
    cache_dir = "data/feature_cache"
    _, _, text_test = load_all_splits("text", cache_dir)
    _, _, audio_test = load_all_splits("audio", cache_dir)
    _, _, video_test = load_all_splits("video", cache_dir)
    
    t_feat_small = text_test[:4]
    a_feat_small = audio_test[:4]
    v_feat_small = video_test[:4]
    
    out = predictor.predict(t_feat_small, a_feat_small, v_feat_small)
    
    for i in range(4):
        print(f"Sample {i}:")
        print(f"  Emotion: {out['emotion']['labels'][i]} (p={out['emotion']['probabilities'][i].max():.4f})")
        print(f"  Sentiment: {out['sentiment']['labels'][i]} (p={out['sentiment']['probabilities'][i].max():.4f})")
        
        # Verify probabilities sum to 1
        emo_sum = out['emotion']['probabilities'][i].sum().item()
        sent_sum = out['sentiment']['probabilities'][i].sum().item()
        assert abs(emo_sum - 1.0) < 1e-5, f"Emotion probabilities sum to {emo_sum}"
        assert abs(sent_sum - 1.0) < 1e-5, f"Sentiment probabilities sum to {sent_sum}"
        
    print("✅ Smoke test passed.")
    
    print("\n==================================================")
    print("3. VERIFY AGAINST FINAL TEST METRICS (FULL SET)")
    print("==================================================")
    
    test_df = load_meld_metadata("data/MELD/annotations/test_sent_emo.csv")
    e_labels = test_df["emotion_id"].values
    s_labels = test_df["sentiment_id"].values
    
    e_preds = []
    s_preds = []
    
    with torch.no_grad():
        for i in range(0, len(text_test), 64):
            end = min(i + 64, len(text_test))
            t_batch = text_test[i:end]
            a_batch = audio_test[i:end]
            v_batch = video_test[i:end]
            
            res = predictor.predict(t_batch, a_batch, v_batch)
            e_preds.extend(res['emotion']['indices'])
            s_preds.extend(res['sentiment']['indices'])
            
    emo_acc = accuracy_score(e_labels, e_preds)
    emo_p_w, emo_r_w, emo_f_w, _ = precision_recall_fscore_support(e_labels, e_preds, average='weighted', zero_division=0)
    emo_p_m, emo_r_m, emo_f_m, _ = precision_recall_fscore_support(e_labels, e_preds, average='macro', zero_division=0)
    
    sent_acc = accuracy_score(s_labels, s_preds)
    sent_p_w, sent_r_w, sent_f_w, _ = precision_recall_fscore_support(s_labels, s_preds, average='weighted', zero_division=0)
    sent_p_m, sent_r_m, sent_f_m, _ = precision_recall_fscore_support(s_labels, s_preds, average='macro', zero_division=0)
    
    print(f"Emotion Accuracy:      {emo_acc*100:.2f}% (Expected: ~55.29%)")
    print(f"Emotion Weighted F1:   {emo_f_w*100:.2f}% (Expected: ~56.03%)")
    print(f"Emotion Macro F1:      {emo_f_m*100:.2f}% (Expected: ~37.01%)")
    print(f"Sentiment Accuracy:    {sent_acc*100:.2f}% (Expected: ~67.16%)")
    print(f"Sentiment Weighted F1: {sent_f_w*100:.2f}% (Expected: ~67.01%)")
    print(f"Sentiment Macro F1:    {sent_f_m*100:.2f}% (Expected: ~64.16%)")
    
    # Assert consistency
    assert abs(emo_acc - 0.5529) < 0.001
    assert abs(emo_f_w - 0.5603) < 0.001
    assert abs(emo_f_m - 0.3701) < 0.001
    assert abs(sent_acc - 0.6716) < 0.001
    assert abs(sent_f_w - 0.6701) < 0.001
    assert abs(sent_f_m - 0.6416) < 0.001
    
    print("\n✅ Metric consistency passed perfectly!")
    
if __name__ == "__main__":
    main()
