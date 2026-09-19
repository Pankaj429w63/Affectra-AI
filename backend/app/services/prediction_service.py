import os
import sys
import torch
from typing import Optional, Dict, Any

# We need to add the project root to sys.path to import the model safely
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.src.inference_model import AffectraPredictor
from backend.app.core.config import settings

class PredictionService:
    _instance: Optional["PredictionService"] = None
    
    def __init__(self):
        self.predictor = None
        self.is_loaded = False
        
    @classmethod
    def get_instance(cls) -> "PredictionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self):
        """Loads the AffectraPredictor model once."""
        if self.is_loaded:
            return
            
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../", settings.MODEL_DIR))
        print(f"Loading AffectraPredictor from {model_dir}...")
        
        try:
            # We enforce CPU device for backend default
            self.predictor = AffectraPredictor(model_dir=model_dir, device="cpu")
            self.is_loaded = True
            print("AffectraPredictor loaded successfully.")
        except Exception as e:
            print(f"Failed to load AffectraPredictor: {e}")
            raise e
            
    def predict(self, text_feat: list, audio_feat: list, video_feat: list) -> Dict[str, Any]:
        """
        Runs inference on given feature vectors.
        Auto-loads model if not already loaded.
        """
        if not self.is_loaded or self.predictor is None:
            self.load_model()
            
        # Convert to torch tensors and add batch dimension
        t_tensor = torch.tensor([text_feat], dtype=torch.float32)
        a_tensor = torch.tensor([audio_feat], dtype=torch.float32)
        v_tensor = torch.tensor([video_feat], dtype=torch.float32)
        
        result = self.predictor.predict(t_tensor, a_tensor, v_tensor)
        
        # Format to match our Pydantic response schema
        # The result returns lists, since batch_size=1, we take index 0
        
        emotion_probs_dict = {
            self.predictor.emotion_labels[i]: result['emotion']['probabilities'][0][i].item()
            for i in range(len(self.predictor.emotion_labels))
        }
        
        sentiment_probs_dict = {
            self.predictor.sentiment_labels[i]: result['sentiment']['probabilities'][0][i].item()
            for i in range(len(self.predictor.sentiment_labels))
        }
        
        return {
            "emotion": {
                "label": result['emotion']['labels'][0],
                "probabilities": emotion_probs_dict
            },
            "sentiment": {
                "label": result['sentiment']['labels'][0],
                "probabilities": sentiment_probs_dict
            }
        }

# Global instance for easy access
prediction_service = PredictionService.get_instance()
