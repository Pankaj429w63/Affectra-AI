import os

class Settings:
    PROJECT_NAME: str = "Affectra AI API"
    VERSION: str = "1.0.0"
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models/affectra_multimodal")
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mock-model")

    # RAG Settings
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RAG_INDEX_DIR: str = os.getenv("RAG_INDEX_DIR", "rag/vectorstore/data")

settings = Settings()
