# Affectra Multimodal Production Model (Experiment 2)

This is the final, validated Experiment 2 champion model for the Affectra-AI project.

## 1. Overview
The ML model predicts emotion and sentiment from pre-extracted multimodal features.
It is a **classification-only** model. The future LLM layer is responsible for natural-language explanations and conversation. Do NOT replace this classifier with an LLM.

## 2. Architecture details
- **Architecture:** Text + Audio + Video (`GatedMultimodalFusion`)
- **Feature Dimensions:** 768 per modality
- **Fusion Dimension:** 512
- **Emotion Classes:** 7 (`neutral`, `joy`, `surprise`, `anger`, `sadness`, `disgust`, `fear`)
- **Sentiment Classes:** 3 (`positive`, `neutral`, `negative`)
- **Best Epoch:** 20

## 3. Final Test Metrics (MELD Test Set)
- **Emotion Accuracy:** 55.29%
- **Emotion Weighted F1:** 56.03%
- **Emotion Macro F1:** 37.01%
- **Sentiment Accuracy:** 67.16%
- **Sentiment Weighted F1:** 67.01%
- **Sentiment Macro F1:** 64.16%

## 4. Usage

To load the model for inference, use the `AffectraPredictor` wrapper class provided in `training/src/inference_model.py`.

```python
import torch
from training.src.inference_model import AffectraPredictor

# 1. Initialize predictor pointing to this directory
predictor = AffectraPredictor(model_dir="models/affectra_multimodal", device="cpu")

# 2. Prepare inputs (these must be [B, 768] feature embeddings extracted by the preprocessing pipeline)
text_features = torch.randn(4, 768)
audio_features = torch.randn(4, 768)
video_features = torch.randn(4, 768)

# 3. Predict
results = predictor.predict(text_features, audio_features, video_features)

print("Emotions:", results["emotion"]["labels"])
print("Sentiments:", results["sentiment"]["labels"])
```

## 5. Input Requirements
The model requires pre-computed 768-dimensional feature embeddings for all three modalities.
- **Text:** DistilRoBERTa (`distilroberta-base`) `[CLS]` token embedding.
- **Audio:** Wav2Vec2 mean-pooled representation.
- **Video:** ViT (`vit-base-patch16-224`) `[CLS]` token embedding.

No raw text, waveform, or raw video is accepted.
