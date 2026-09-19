from fastapi import APIRouter, HTTPException
from backend.app.schemas.explanation import ExplanationRequest, ExplanationResponse
from backend.app.services.explanation_service import explanation_service

router = APIRouter()

@router.post("/explain", response_model=ExplanationResponse, summary="Explain Emotion and Sentiment")
async def explain(request: ExplanationRequest):
    try:
        result = explanation_service.generate_explanation(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")
