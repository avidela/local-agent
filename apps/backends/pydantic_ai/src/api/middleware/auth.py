"""
Authentication middleware for JWT token validation and user context injection.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...database.models import User
from ...services.auth.jwt_service import get_jwt_service


# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from JWT token (optional - returns None if no token or invalid).
    
    Args:
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        User object if valid token, None otherwise
    """
    if not credentials:
        return None
    
    jwt_service = get_jwt_service()
    user = await jwt_service.get_current_user(db, credentials.credentials)
    return user


async def get_current_user_required(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current user from JWT token (required - raises exception if no valid token).
    
    Args:
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If no valid token provided
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    jwt_service = get_jwt_service()
    user = await jwt_service.get_current_user(db, credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user_required)
) -> User:
    """
    Get current active user (must be authenticated and active).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def require_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Require current user to be an admin.
    
    Args:
        current_user: Current authenticated active user
        
    Returns:
        Admin user object
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user