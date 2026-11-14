"""Authentication dependency and API key validation."""

import logging
from typing import Optional
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import User
from backend.database.repository import UserRepository
from backend.database.session import get_db

logger = logging.getLogger(__name__)

api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    x_api_key: Optional[str] = Security(api_key_header_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Authenticates API requests via X-API-Key or Bearer token."""
    if not settings.ENABLE_AUTH:
        # Development mode bypass
        guest_user = db.query(User).first()
        if guest_user is not None:
            return guest_user
        return User(id=1, username="dev-guest", is_active=True)

    extracted_key = None
    if x_api_key:
        extracted_key = x_api_key.strip()
    elif bearer_creds and bearer_creds.credentials:
        extracted_key = bearer_creds.credentials.strip()

    if not extracted_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_CREDENTIALS",
                "message": "Authentication required. Provide 'X-API-Key' header or 'Bearer <token>'.",
            },
        )

    # 1. Fast match master key from settings
    if extracted_key == settings.API_KEY:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            return admin_user
        return User(id=1, username="admin", is_active=True)

    # 2. Check hashed database users
    user = UserRepository.verify_api_key(db, extracted_key)
    if not user:
        logger.warning("Unauthorized access attempt with invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY",
                "message": "The provided API key is invalid or deactivated.",
            },
        )

    return user
