from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Optional
from backend.app.schemas.prediction import PredictionRequest
from backend.app.schemas.response import PredictionResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.feature_extraction_service import feature_extraction_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, summary="Predict Emotion and Sentiment from pre-extracted features")
async def predict(request: PredictionRequest):
    if not prediction_service.is_loaded:
        try:
            prediction_service.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
        
    try:
        result = prediction_service.predict(
            text_feat=request.text_feat,
            audio_feat=request.audio_feat,
            video_feat=request.video_feat
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.post("/predict/raw", response_model=PredictionResponse, summary="Extract features and predict from raw inputs")
async def predict_raw(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
):
    if not prediction_service.is_loaded:
        try:
            prediction_service.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")

    has_text = bool(text and text.strip())
    has_audio = bool(audio_file and audio_file.filename)
    has_video = bool(video_file and video_file.filename)

    if not (has_text or has_audio or has_video):
        raise HTTPException(status_code=400, detail="At least one modality (text, audio, or video) must be provided.")

    try:
        audio_bytes = await audio_file.read() if has_audio else None
        video_bytes = await video_file.read() if has_video else None

        # Text features
        text_feat = feature_extraction_service.extract_text_feature(text if has_text else None)

        # Audio features
        if has_audio:
            audio_feat = feature_extraction_service.extract_audio_feature(audio_bytes, audio_file.filename)
        elif has_video:
            # Extract audio track from uploaded video if audio_file not provided
            audio_feat = feature_extraction_service.extract_audio_feature(video_bytes, video_file.filename)
        else:
            audio_feat = [0.0] * 768

        # Video features
        video_feat = feature_extraction_service.extract_video_feature(video_bytes, video_file.filename if has_video else None)

        # Run model prediction with extracted 768-d features
        result = prediction_service.predict(
            text_feat=text_feat,
            audio_feat=audio_feat,
            video_feat=video_feat
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal prediction failed: {str(e)}")

