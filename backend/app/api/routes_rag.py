from fastapi import APIRouter, HTTPException
from backend.app.schemas.rag import RAGRequest, RAGResponse
from backend.app.services.rag_service import rag_service

router = APIRouter()

@router.post("/rag", response_model=RAGResponse, summary="Retrieve and Answer using RAG")
async def rag_endpoint(request: RAGRequest):
    try:
        result = rag_service.generate_answer(request)
        return result
    except ValueError as ve:
        # For validation errors like invalid top_k passed internally, 
        # though Pydantic should catch most of these.
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the RAG request.")
