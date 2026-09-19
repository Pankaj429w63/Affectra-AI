from pydantic import BaseModel, Field, validator
from typing import List

class RAGRequest(BaseModel):
    question: str = Field(..., description="The user's question to answer using the Affectra knowledge base.")
    top_k: int = Field(3, ge=1, description="The maximum number of context chunks to retrieve.")

    @validator('question')
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question must not be empty or whitespace-only.')
        return v.strip()

class RAGContextItem(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: List[RAGContextItem]
