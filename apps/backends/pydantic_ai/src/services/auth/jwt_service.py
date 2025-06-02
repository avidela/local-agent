"""
JWT service for token generation, validation, and password management.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...database.models import User


class JWTService:
    """Service for JWT token management and password hashing."""
    
    def __init__(self):
        """Initialize JWT service with password context."""
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__memory_cost=65536,  # 64 MB
            argon2__time_cost=3,        # 3 iterations
            argon2__parallelism=2       # 2 threads
        )
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Payload data to encode in the token
            expires_delta: Custom expiration time, defaults to settings value
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.auth.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.auth.secret_key.get_secret_value(),
            algorithm=settings.auth.algorithm
        )
        
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string to decode
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.auth.secret_key.get_secret_value(),
                algorithms=[settings.auth.algorithm]
            )
            return payload
        except JWTError:
            return None
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using argon2.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password to check against
            
        Returns:
            True if password matches, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def authenticate_user(
        self, 
        db: AsyncSession, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """
        Authenticate a user by username and password.
        
        Args:
            db: Database session
            username: Username to authenticate
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        # Query user by username
        result = await db.execute(
            select(User).where(User.username == username, User.is_active == True)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Update last login time
        user.last_login = datetime.utcnow()
        await db.commit()
        
        return user
    
    def create_user_token_data(self, user: User) -> Dict[str, Any]:
        """
        Create token data payload for a user.
        
        Args:
            user: User object to create token for
            
        Returns:
            Token payload dictionary
        """
        # Handle role - could be enum object or string from database
        role_value = user.role.value if hasattr(user.role, 'value') else user.role
        
        return {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": role_value,
            "is_active": user.is_active,
        }
    
    async def get_current_user(
        self,
        db: AsyncSession,
        token: str
    ) -> Optional[User]:
        """
        Get current user from JWT token.
        
        Args:
            db: Database session
            token: JWT token string
            
        Returns:
            User object if token is valid, None otherwise
        """
        payload = self.decode_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Handle admin fallback case (when sub="admin")
        if user_id == "admin":
            # Create a mock admin user object for the session
            from ...database.models import UserRole
            from datetime import datetime
            now = datetime.utcnow()
            mock_admin = User(
                id=0,  # Special admin ID
                username=payload.get("username", "admin"),
                email=payload.get("email", "admin@localhost"),
                password_hash="",  # Not used for admin tokens
                role=UserRole.ADMIN,
                is_active=True,
                created_at=now,
                updated_at=now,
                last_login=now
            )
            return mock_admin
        
        try:
            # Query user by ID for regular database users
            result = await db.execute(
                select(User).where(User.id == int(user_id), User.is_active == True)
            )
            user = result.scalar_one_or_none()
            return user
        except (ValueError, Exception):
            return None
    
    async def create_user(
        self,
        db: AsyncSession,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        role: str = "user"
    ) -> Optional[User]:
        """
        Create a new user with hashed password.
        
        Args:
            db: Database session
            username: Username for the new user
            email: Email for the new user
            password: Plain text password (will be hashed)
            full_name: Optional full name
            role: User role (default: "user")
            
        Returns:
            Created User object or None if creation failed
        """
        try:
            # Check if username or email already exists
            result = await db.execute(
                select(User).where(
                    (User.username == username) | (User.email == email)
                )
            )
            existing_user = result.scalar_one_or_none()
            if existing_user:
                return None  # User already exists
            
            # Hash the password
            hashed_password = self.hash_password(password)
            
            # Create new user
            from ...database.models import UserRole
            from datetime import datetime
            
            new_user = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                full_name=full_name,
                role=role,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            
            return new_user
            
        except Exception:
            await db.rollback()
            return None
    
    async def change_password(
        self,
        db: AsyncSession,
        user_id: int,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password after verifying current password.
        
        Args:
            db: Database session
            user_id: ID of the user
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            True if password changed successfully, False otherwise
        """
        try:
            # Get user
            result = await db.execute(
                select(User).where(User.id == user_id, User.is_active == True)
            )
            user = result.scalar_one_or_none()
            if not user:
                return False
            
            # Verify current password
            if not self.verify_password(current_password, user.password_hash):
                return False
            
            # Hash new password and update
            user.password_hash = self.hash_password(new_password)
            
            await db.commit()
            return True
            
        except Exception:
            await db.rollback()
            return False
    
    async def reset_password(
        self,
        db: AsyncSession,
        username: str,
        new_password: str
    ) -> bool:
        """
        Reset user password (admin function).
        
        Args:
            db: Database session
            username: Username of the user
            new_password: New password to set
            
        Returns:
            True if password reset successfully, False otherwise
        """
        try:
            # Get user by username
            result = await db.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                return False
            
            # Hash new password and update
            user.password_hash = self.hash_password(new_password)
            
            await db.commit()
            return True
            
        except Exception:
            await db.rollback()
            return False
    
    async def toggle_user_status(
        self,
        db: AsyncSession,
        user_id: int,
        is_active: bool
    ) -> bool:
        """
        Enable or disable a user account.
        
        Args:
            db: Database session
            user_id: ID of the user
            is_active: True to enable, False to disable
            
        Returns:
            True if status changed successfully, False otherwise
        """
        try:
            # Get user
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            if not user:
                return False
            
            # Update status
            user.is_active = is_active
            
            await db.commit()
            return True
            
        except Exception:
            await db.rollback()
            return False

    async def list_users(
        self,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0
    ) -> List[User]:
        """
        List all users (admin function).
        
        Args:
            db: Database session
            limit: Maximum number of users to return
            offset: Number of users to skip
            
        Returns:
            List of User objects
        """
        try:
            result = await db.execute(
                select(User)
                .order_by(User.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            users = result.scalars().all()
            return list(users)
            
        except Exception:
            return []


# Global JWT service instance
_jwt_service: Optional[JWTService] = None


def get_jwt_service() -> JWTService:
    """Get the global JWT service instance."""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService()
    return _jwt_service