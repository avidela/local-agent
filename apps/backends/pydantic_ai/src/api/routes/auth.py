"""
Authentication API routes with real JWT implementation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...database import get_db
from ...database.models import User
from ...services.auth.jwt_service import get_jwt_service
from ..middleware.auth import get_current_active_user, require_admin_user
from ..schemas import (
    LoginRequest, Token, UserResponse, RegisterRequest,
    ChangePasswordRequest, ResetPasswordRequest, UserManagementResponse
)

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint with real JWT token generation.
    
    Args:
        login_data: Username and password
        db: Database session
        
    Returns:
        JWT access token with user information
        
    Raises:
        HTTPException: If credentials are invalid
    """
    jwt_service = get_jwt_service()
    
    # First try to authenticate against database users
    user = await jwt_service.authenticate_user(db, login_data.username, login_data.password)
    
    # If no database user found, check against admin credentials as fallback
    if not user and (
        login_data.username == settings.auth.admin_username and
        login_data.password == settings.auth.admin_password.get_secret_value()
    ):
        # Create admin user token data
        token_data = {
            "sub": "admin",
            "username": settings.auth.admin_username,
            "email": "admin@localhost",
            "role": "admin",
            "is_active": True,
        }
    else:
        # Use authenticated user data
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = jwt_service.create_user_token_data(user)
    
    # Generate JWT token
    access_token = jwt_service.create_access_token(data=token_data)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    Refresh JWT token for authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        New JWT access token
    """
    jwt_service = get_jwt_service()
    
    # Create new token data
    token_data = jwt_service.create_user_token_data(current_user)
    
    # Generate new JWT token
    access_token = jwt_service.create_access_token(data=token_data)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.auth.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout():
    """
    Logout endpoint (JWT tokens are stateless, so this is informational).
    
    Returns:
        Success message
    """
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user from JWT token
        
    Returns:
        Current user information
    """
    return UserResponse.model_validate(current_user)


# User Management Endpoints

@router.post("/register", response_model=UserManagementResponse)
async def register_user(
    register_data: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    Args:
        register_data: User registration information
        db: Database session
        
    Returns:
        Registration confirmation with user ID
        
    Raises:
        HTTPException: If username/email already exists
    """
    jwt_service = get_jwt_service()
    
    user = await jwt_service.create_user(
        db=db,
        username=register_data.username,
        email=register_data.email,
        password=register_data.password,
        full_name=register_data.full_name
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    
    return UserManagementResponse(
        message="User registered successfully",
        user_id=user.id
    )


@router.post("/change-password", response_model=UserManagementResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user's password.
    
    Args:
        password_data: Current and new password
        current_user: Currently authenticated user
        db: Database session
        
    Returns:
        Password change confirmation
        
    Raises:
        HTTPException: If current password is incorrect
    """
    jwt_service = get_jwt_service()
    
    success = await jwt_service.change_password(
        db=db,
        user_id=current_user.id,
        current_password=password_data.current_password,
        new_password=password_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return UserManagementResponse(
        message="Password changed successfully",
        user_id=current_user.id
    )


# Admin-only endpoints

@router.post("/admin/reset-password", response_model=UserManagementResponse)
async def admin_reset_password(
    reset_data: ResetPasswordRequest,
    admin_user: User = Depends(require_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Reset a user's password (admin only).
    
    Args:
        reset_data: Username and new password
        admin_user: Admin user performing the action
        db: Database session
        
    Returns:
        Password reset confirmation
        
    Raises:
        HTTPException: If user not found
    """
    jwt_service = get_jwt_service()
    
    success = await jwt_service.reset_password(
        db=db,
        username=reset_data.username,
        new_password=reset_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserManagementResponse(
        message=f"Password reset successfully for user: {reset_data.username}",
        user_id=0  # We don't have the user ID in this context
    )


@router.post("/admin/users/{user_id}/disable", response_model=UserManagementResponse)
async def disable_user(
    user_id: int,
    admin_user: User = Depends(require_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Disable a user account (admin only).
    
    Args:
        user_id: ID of the user to disable
        admin_user: Admin user performing the action
        db: Database session
        
    Returns:
        User disable confirmation
        
    Raises:
        HTTPException: If user not found
    """
    jwt_service = get_jwt_service()
    
    success = await jwt_service.toggle_user_status(
        db=db,
        user_id=user_id,
        is_active=False
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserManagementResponse(
        message="User disabled successfully",
        user_id=user_id
    )


@router.post("/admin/users/{user_id}/enable", response_model=UserManagementResponse)
async def enable_user(
    user_id: int,
    admin_user: User = Depends(require_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Enable a user account (admin only).
    
    Args:
        user_id: ID of the user to enable
        admin_user: Admin user performing the action
        db: Database session
        
    Returns:
        User enable confirmation
        
    Raises:
        HTTPException: If user not found
    """
    jwt_service = get_jwt_service()
    
    success = await jwt_service.toggle_user_status(
        db=db,
        user_id=user_id,
        is_active=True
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserManagementResponse(
        message="User enabled successfully",
        user_id=user_id
    )


@router.get("/admin/users", response_model=List[UserResponse])
async def list_users(
    limit: int = 50,
    offset: int = 0,
    admin_user: User = Depends(require_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users (admin only).
    
    Args:
        limit: Maximum number of users to return
        offset: Number of users to skip
        admin_user: Admin user requesting the list
        db: Database session
        
    Returns:
        List of users
    """
    jwt_service = get_jwt_service()
    
    users = await jwt_service.list_users(
        db=db,
        limit=limit,
        offset=offset
    )
    
    return [UserResponse.model_validate(user) for user in users]