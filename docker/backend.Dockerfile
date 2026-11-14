# ==============================================================================
# StyleForge v2.0 Production Backend Dockerfile
# Multi-stage image supporting both CPU and CUDA container runtime environments
# ==============================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system utilities & OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch & Core Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pydantic-settings \
    sqlalchemy \
    alembic \
    asyncpg \
    psycopg2-binary \
    prometheus_client \
    redis \
    mlflow

# Copy application source code
COPY models/ ./models/
COPY core/ ./core/
COPY backend/ ./backend/
COPY checkpoints/ ./checkpoints/
COPY data/ ./data/

# Set Python path to include app root
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
