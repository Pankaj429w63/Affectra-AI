FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies (e.g., for FAISS or OpenCV if needed by encoders)
# We install libgomp1 which is often needed by FAISS on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy required production code and models
# We copy specific directories to avoid unnecessary files
COPY backend/ /app/backend/
COPY training/src/ /app/training/src/
COPY rag/ /app/rag/
COPY models/affectra_multimodal/ /app/models/affectra_multimodal/

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
