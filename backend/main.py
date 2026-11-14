"""StyleForge Production REST API Application Entry Point."""

from contextlib import asynccontextmanager
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST

from backend.api.v1.router import api_v1_router
from backend.config import settings
from backend.database.session import init_db
from backend.metrics.prometheus import get_metrics_output
from backend.schemas.responses import HealthResponse
from core.engine.checkpoint_manager import CheckpointManager
from core.engine.device_manager import DeviceManager
from core.engine.inference_engine import InferenceEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("styleforge.backend")

start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management for startup initialization and shutdown cleanup."""
    logger.info("Initializing StyleForge Backend (v%s, env=%s)...", settings.APP_VERSION, settings.ENV)

    # 1. Initialize DB schema and seeds
    try:
        init_db()
    except Exception as e:
        logger.error("Database initialization failed: %s", e, exc_info=True)

    # 2. Warm up Model in Inference Engine
    try:
        logger.info("Pre-loading model from %s...", settings.CHECKPOINT_PATH)
        engine = InferenceEngine.get_instance(checkpoint_path=settings.CHECKPOINT_PATH, device=settings.DEVICE)
        logger.info("Model pre-loaded successfully on device %s", engine.device)
    except Exception as e:
        logger.error("Failed to load production checkpoint on startup: %s", e, exc_info=True)

    yield

    # Shutdown cleanup
    logger.info("Shutting down StyleForge Backend. Flushing GPU caches...")
    DeviceManager.empty_cache()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-Grade Generative AI Platform for StyleForge Facial Synthesis.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pure ASGI Middleware for Request ID & Process Time
class RequestContextMiddleware:
    """Zero-overhead ASGI middleware injecting correlation IDs and request timing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_id = str(uuid.uuid4())
        start_t = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", req_id.encode("latin1")))
                headers.append((b"x-process-time-ms", f"{duration_ms:.2f}".encode("latin1")))
                message["headers"] = headers
            await send(message)

        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = req_id
        await self.app(scope, receive, send_wrapper)


app.add_middleware(RequestContextMiddleware)


# Global Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", "unknown")
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    code = detail.get("code", f"HTTP_{exc.status_code}")
    message = detail.get("message", "An error occurred.")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": req_id,
                **{k: v for k, v in detail.items() if k not in ("code", "message")},
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception for request %s: %s", req_id, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred.",
                "request_id": req_id,
            }
        },
    )


# System Endpoints
@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Service health check")
def health_check():
    """Returns application health and runtime telemetry."""
    uptime = time.time() - start_time
    engine = InferenceEngine.get_instance(checkpoint_path=settings.CHECKPOINT_PATH, device=settings.DEVICE)
    cuda_avail = DeviceManager.resolve_device().type == "cuda"

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        version=settings.APP_VERSION,
        device=str(engine.device),
        cuda_available=cuda_avail,
        database_connected=True,
        models_loaded=engine.raw_generator is not None,
    )


@app.get("/ready", tags=["system"], summary="Kubernetes / Docker readiness probe")
def readiness_check():
    """Readiness probe indicating if service is ready to accept traffic."""
    engine = InferenceEngine.get_instance(checkpoint_path=settings.CHECKPOINT_PATH, device=settings.DEVICE)
    if engine.raw_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MODEL_NOT_READY", "message": "Model weights are not loaded."},
        )
    return {"status": "ready"}


@app.get("/api/v1/metrics", tags=["observability"], summary="Prometheus scrape endpoint")
def prometheus_metrics():
    """Exports system and inference telemetry in Prometheus exposition format."""
    payload = get_metrics_output()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


# Mount API v1 Routes
app.include_router(api_v1_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
