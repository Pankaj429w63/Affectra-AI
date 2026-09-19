import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath("."))

from training.src.config import EMOTION_LABELS, SENTIMENT_LABELS

def main():
    src_ckpt = "data/exp2_checkpoints/checkpoint_best.pt"
    out_dir = "models/affectra_multimodal"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load original checkpoint
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    
    # 2. Extract state dict
    model_state = ckpt.get("model_state_dict", ckpt)
    
    # 3. Save production model.pt
    torch.save(model_state, os.path.join(out_dir, "model.pt"))
    
    # 4. Save labels.json
    labels_dict = {
        "emotions": {i: label for i, label in enumerate(EMOTION_LABELS)},
        "sentiments": {i: label for i, label in enumerate(SENTIMENT_LABELS)}
    }
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels_dict, f, indent=4)
        
    # 5. Save config
    config = {
        "model_name": "AffectraMultimodal",
        "experiment": "Experiment 2",
        "fusion_dim": 512,
        "text_feature_dim": 768,
        "audio_feature_dim": 768,
        "video_feature_dim": 768,
        "num_emotions": 7,
        "num_sentiments": 3,
        "emotion_labels": EMOTION_LABELS,
        "sentiment_labels": SENTIMENT_LABELS,
        "checkpoint_source": src_ckpt,
        "best_epoch": ckpt.get("epoch", 20),
        "test_emotion_accuracy": 0.5529,
        "test_emotion_weighted_f1": 0.5603,
        "test_emotion_macro_f1": 0.3701,
        "test_sentiment_accuracy": 0.6716,
        "test_sentiment_weighted_f1": 0.6701,
        "test_sentiment_macro_f1": 0.6416
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Exported model to {out_dir}/model.pt")
    
if __name__ == "__main__":
    main()
