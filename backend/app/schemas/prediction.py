from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    # Expecting arrays of floats representing the embeddings
    # The default dimension for Experiment 2 models is 768 for each modality
    text_feat: List[float] = Field(..., min_length=768, max_length=768, description="Text feature embeddings, expected length 768")
    audio_feat: List[float] = Field(..., min_length=768, max_length=768, description="Audio feature embeddings, expected length 768")
    video_feat: List[float] = Field(..., min_length=768, max_length=768, description="Video feature embeddings, expected length 768")
