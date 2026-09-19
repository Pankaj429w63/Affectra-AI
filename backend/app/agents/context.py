from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class AgentContext(BaseModel):
    """
    Shared context passed between agents.
    Holds minimal necessary state to avoid massive global state.
    """
    # Original user query (if applicable)
    user_query: Optional[str] = None
    
    # ML Prediction inputs
    emotion_label: Optional[str] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    sentiment_label: Optional[str] = None
    sentiment_probabilities: Optional[Dict[str, float]] = None
    
    # RAG parameters
    top_k: int = 3
    
    # Agent Outputs
    interpretation_result: Optional[str] = None
    retrieved_context_str: Optional[str] = None
    retrieved_context_items: Optional[List[Dict[str, Any]]] = None
    response_draft: Optional[str] = None
    
    # Final Output
    final_response: Optional[str] = None
    safety_approved: bool = True
    
    # Execution Trace
    execution_trace: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
