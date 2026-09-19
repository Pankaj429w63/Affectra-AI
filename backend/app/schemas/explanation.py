from pydantic import BaseModel, Field
from typing import Dict

class ExplanationRequest(BaseModel):
    emotion_label: str = Field(..., description="The predicted emotion label")
    emotion_probabilities: Dict[str, float] = Field(..., description="Probabilities of all emotion classes")
    sentiment_label: str = Field(..., description="The predicted sentiment label")
    sentiment_probabilities: Dict[str, float] = Field(..., description="Probabilities of all sentiment classes")

class ExplanationResponse(BaseModel):
    explanation: str = Field(..., description="Natural language explanation of the ML prediction")
