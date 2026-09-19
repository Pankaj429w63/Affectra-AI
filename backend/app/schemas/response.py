from pydantic import BaseModel
from typing import Dict, Any

class EmotionResult(BaseModel):
    label: str
    probabilities: Dict[str, float]

class SentimentResult(BaseModel):
    label: str
    probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    emotion: EmotionResult
    sentiment: SentimentResult
