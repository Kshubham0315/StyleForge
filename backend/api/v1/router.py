"""API v1 master router aggregating sub-modules."""

from fastapi import APIRouter
from backend.api.v1.generate import router as generate_router
from backend.api.v1.history import router as history_router
from backend.api.v1.models import router as models_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(generate_router)
api_v1_router.include_router(history_router)
api_v1_router.include_router(models_router)
