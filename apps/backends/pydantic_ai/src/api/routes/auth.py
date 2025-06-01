"""
Authentication API routes.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...database import get_db
from ..schemas import LoginRequest, Token

router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint to get JWT token.
    
    Args:
        login_data: Username and password
        db: Database session
        
    Returns:
        JWT access token
    """
    
    # For now, simple admin authentication
    # In production, this would verify against the User table with proper password hashing
    if (login_data.username == settings.auth.admin_username and 
        login_data.password == settings.auth.admin_password.get_secret_value()):
        
        return Token(
            access_token="demo-jwt-token",  # In production, generate real JWT
            token_type="bearer",
            expires_in=settings.auth.access_token_expire_minutes * 60,
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/logout")
async def logout():
    """
    Logout endpoint (for consistency, JWT tokens are stateless).
    
    Returns:
        Success message
    """
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_current_user(token: str = Depends(security)):
    """
    Get current user information.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Current user information
    """
    
    # In production, decode and validate JWT token
    return {
        "username": settings.auth.admin_username,
        "role": "admin",
        "is_active": True,
    }