from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.services.prediction_service import prediction_service

router = APIRouter()

@router.get("/health", summary="Health Check")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "model_loaded": prediction_service.is_loaded
    }
