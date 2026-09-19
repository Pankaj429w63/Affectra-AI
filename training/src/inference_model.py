import os
import json
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Union

from training.src.fusion_model import GatedMultimodalFusion

class AffectraPredictor:
    """
    Production inference wrapper for the Affectra-AI multimodal model (Experiment 2).
    """
    def __init__(self, model_dir: str, device: Union[str, torch.device] = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Load config and labels
        config_path = os.path.join(model_dir, "model_config.json")
        labels_path = os.path.join(model_dir, "labels.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file missing: {config_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file missing: {labels_path}")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
            self.emotion_labels = {int(k): v for k, v in labels_data["emotions"].items()}
            self.sentiment_labels = {int(k): v for k, v in labels_data["sentiments"].items()}
            
        # Reconstruct exactly the Experiment 2 architecture
        self.model = GatedMultimodalFusion(
            input_dim=self.config.get("text_feature_dim", 768),
            fusion_dim=self.config.get("fusion_dim", 512),
            num_emotions=self.config.get("num_emotions", 7),
            num_sentiments=self.config.get("num_sentiments", 3),
            dropout=0.0, # Dropout doesn't matter for inference since we call eval()
            modality_dropout=0.0
        ).to(self.device)
        
        # Load weights
        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights missing: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        """
        Run inference on pre-extracted features.
        
        Args:
            text_feat (torch.Tensor): [B, 768]
            audio_feat (torch.Tensor): [B, 768]
            video_feat (torch.Tensor): [B, 768]
            
        Returns:
            Dict containing predicted labels, logits, and probabilities for both emotion and sentiment.
        """
        # Move inputs to device
        text_feat = text_feat.to(self.device)
        audio_feat = audio_feat.to(self.device)
        video_feat = video_feat.to(self.device)
        
        # Create masks (all modalities present)
        b_size = text_feat.size(0)
        t_mask = torch.ones(b_size, device=self.device)
        a_mask = torch.ones(b_size, device=self.device)
        v_mask = torch.ones(b_size, device=self.device)
        
        # Forward pass
        emo_logits, sent_logits = self.model(
            text_feat, audio_feat, video_feat,
            t_mask, a_mask, v_mask
        )
        
        # Probabilities
        emo_probs = torch.softmax(emo_logits, dim=1)
        sent_probs = torch.softmax(sent_logits, dim=1)
        
        # Class indices
        emo_preds = torch.argmax(emo_logits, dim=1).cpu().tolist()
        sent_preds = torch.argmax(sent_logits, dim=1).cpu().tolist()
        
        # Map to human-readable strings
        emo_labels_pred = [self.emotion_labels[idx] for idx in emo_preds]
        sent_labels_pred = [self.sentiment_labels[idx] for idx in sent_preds]
        
        return {
            "emotion": {
                "labels": emo_labels_pred,
                "indices": emo_preds,
                "logits": emo_logits.cpu(),
                "probabilities": emo_probs.cpu()
            },
            "sentiment": {
                "labels": sent_labels_pred,
                "indices": sent_preds,
                "logits": sent_logits.cpu(),
                "probabilities": sent_probs.cpu()
            }
        }
