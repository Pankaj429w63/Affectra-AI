from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.core.config import settings
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_predict import router as predict_router
from backend.app.api.routes_explain import router as explain_router
from backend.app.api.routes_rag import router as rag_router
from backend.app.api.routes_agents import router as agents_router
from backend.app.services.prediction_service import prediction_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    print("Starting up Affectra AI API...")
    try:
        prediction_service.load_model()
    except Exception as e:
        print(f"Error loading model during startup: {e}")
        # Not exiting so /health can report model_loaded: false if needed for debugging
        
    try:
        from backend.app.services.rag_service import rag_service
        rag_service.load()
    except Exception as e:
        print(f"Error loading RAG service during startup: {e}")

    yield
    # Cleanup on shutdown
    print("Shutting down Affectra AI API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "https://affectra-ai.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["Prediction"])
app.include_router(explain_router, tags=["Explanation"])
app.include_router(rag_router, tags=["RAG"])
app.include_router(agents_router, tags=["Agents"])

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

