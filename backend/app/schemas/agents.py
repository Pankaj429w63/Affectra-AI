from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AgentAPIRequest(BaseModel):
    user_query: Optional[str] = Field(None, description="The user's question, if any.")
    emotion_label: Optional[str] = Field(None, description="The predicted emotion label.")
    emotion_probabilities: Optional[Dict[str, float]] = Field(None, description="The probabilities of all emotion classes.")
    sentiment_label: Optional[str] = Field(None, description="The predicted sentiment label.")
    sentiment_probabilities: Optional[Dict[str, float]] = Field(None, description="The probabilities of all sentiment classes.")
    top_k: int = Field(3, ge=1, le=100, description="Number of knowledge base items to retrieve.")

class AgentContextItem(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str

class AgentAPIResponse(BaseModel):
    final_response: str = Field(..., description="The final drafted response from the agent pipeline.")
    safety_approved: bool = Field(..., description="Whether the response passed safety guardrails.")
    execution_trace: List[str] = Field(..., description="The ordered sequence of executed agents.")
    retrieved_context: List[AgentContextItem] = Field(default_factory=list, description="Any contextual knowledge retrieved by the Knowledge Agent.")
    error_message: Optional[str] = Field(None, description="An error message if the pipeline failed.")
