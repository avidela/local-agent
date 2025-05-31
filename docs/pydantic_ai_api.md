# 🌐 API Endpoints & FastAPI Integration

> **Complete REST API with streaming support using FastAPI and Pydantic AI**

## 🎯 Overview

The API layer provides RESTful endpoints for agent management, session handling, and real-time streaming using FastAPI. All endpoints follow RESTful conventions with proper authentication and error handling.

## 🚀 Main FastAPI Application

### Application Setup with Dependencies
```python
# src/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os

from .database.connection import init_db
from .api.v1 import agents, sessions, marketplace, users
from .observability.instrumentation import observability

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with initialization"""
    
    # Initialize observability
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        observability.configure_logfire(
            token=os.getenv("LOGFIRE_TOKEN"),
            send_to_logfire=True,
            event_mode='events'
        )
    elif environment == "staging":
        observability.configure_alternative_otel_backend(
            endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318"),
            event_mode='logs'
        )
    else:
        if os.getenv("OTEL_ENDPOINT"):
            observability.configure_raw_opentelemetry(
                endpoint=os.getenv("OTEL_ENDPOINT")
            )
    
    # Initialize database
    await init_db()
    
    yield
    
    # Cleanup

def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Pydantic AI Agents Service",
        description="Production-ready AI agents service with conversation history",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if os.getenv("DEBUG") == "true" else "An error occurred"
            }
        )
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "pydantic-ai-agents",
            "version": "1.0.0"
        }
    
    # Include routers
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(marketplace.router, prefix="/api/v1")
    
    return app

app = create_app()
```

## 👥 User Management API

### User Authentication and Management
```python
# src/api/v1/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from ...database.session import get_db
from ...services.user_service import UserService
from ...schemas.users import UserCreate, UserResponse, UserLogin, TokenResponse
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/users", tags=["users"])
security = HTTPBearer()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    
    user_service = UserService()
    
    try:
        user = await user_service.create_user(user_data, db)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT token"""
    
    user_service = UserService()
    
    try:
        token_data = await user_service.authenticate_user(
            credentials.username, credentials.password, db
        )
        return token_data
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get current user information"""
    
    user_service = UserService()
    
    try:
        user = await user_service.get_user_by_id(user_id, db)
        return user
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    updates: UserUpdate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information"""
    
    user_service = UserService()
    
    try:
        user = await user_service.update_user(user_id, updates, db)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
```

## 🤖 Agents API

### Agent Management Endpoints
```python
# src/api/v1/agents.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from ...database.session import get_db
from ...services.agent_service import AgentService
from ...services.model_service import ModelService
from ...schemas.agents import AgentCreate, AgentResponse, AgentUpdate, ToolInfo
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Create a new agent"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    try:
        agent = await agent_service.create_agent(agent_data, user_id, db)
        return agent
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    user_id: UUID = Depends(get_current_user_id),
    include_public: bool = Query(False, description="Include public agents"),
    db: AsyncSession = Depends(get_db)
):
    """List user's agents"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    agents = await agent_service.list_user_agents(user_id, db, include_public)
    return agents

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get specific agent details"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    try:
        # This would include permission checking
        agent_model = await agent_service._get_agent_model(agent_id, db)
        
        # Check if user has access (owner or public)
        if agent_model.owner_id != user_id and not agent_model.is_public:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return AgentResponse.from_orm(agent_model)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: UUID,
    updates: AgentUpdate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Update agent configuration"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    try:
        agent = await agent_service.update_agent(
            agent_id, updates.dict(exclude_unset=True), user_id, db
        )
        return agent
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Delete agent"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    try:
        await agent_service.delete_agent(agent_id, user_id, db)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/tools/available", response_model=List[str])
async def get_available_tools():
    """Get list of available tools"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    return agent_service.get_available_tools()

@router.post("/{agent_id}/test")
async def test_agent(
    agent_id: UUID,
    test_message: str = Query(..., description="Test message to send"),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Test agent with a simple message"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    
    try:
        # Get agent instance
        agent = await agent_service.get_agent_instance(agent_id, db)
        
        # Run simple test
        result = await agent.run(test_message)
        
        return {
            "success": True,
            "response": result.data,
            "cost": result.cost(),
            "message_count": len(result.all_messages())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

## 💬 Sessions API

### Session and Message Management
```python
# src/api/v1/sessions.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import json

from ...database.session import get_db
from ...services.session_service import SessionService
from ...services.agent_service import AgentService
from ...services.model_service import ModelService
from ...schemas.sessions import (
    SessionCreate, SessionResponse, MessageCreate, EventResponse,
    ConversationHistory, SessionStats
)
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Create a new session"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        session = await session_service.create_session(session_data, user_id, db)
        return session
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    user_id: UUID = Depends(get_current_user_id),
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """List user's sessions"""
    
    # Implementation would include filtering and pagination
    from sqlalchemy import select, desc
    from ...models.session import Session
    
    query = select(Session).where(Session.user_id == user_id)
    
    if agent_id:
        query = query.where(Session.agent_id == agent_id)
    
    query = query.order_by(desc(Session.updated_at)).limit(limit).offset(offset)
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return [SessionResponse.from_orm(session) for session in sessions]

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get session details"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    session = await session_service._get_session_with_history(session_id, user_id, db)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return SessionResponse.from_orm(session)

@router.post("/{session_id}/messages")
async def send_message(
    session_id: UUID,
    message: MessageCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Send message to agent (streaming or batch)"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        if message.streaming:
            # Streaming response using Server-Sent Events
            async def event_stream():
                async for event in session_service.send_message(session_id, message, user_id, db):
                    # Format as Server-Sent Events
                    yield f"data: {event.json()}\n\n"
                
                # Send end-of-stream marker
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                event_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Batch response
            events = []
            async for event in session_service.send_message(session_id, message, user_id, db):
                events.append(event)
            
            return events[0] if events else None
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{session_id}/events", response_model=List[EventResponse])
async def get_session_events(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Get events for a session"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        events = await session_service.get_session_events(
            session_id, user_id, db, limit, offset
        )
        return events
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.get("/{session_id}/history", response_model=List[dict])
async def get_conversation_history(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get conversation history for session"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        history = await session_service.get_conversation_history(session_id, user_id, db)
        return history
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.post("/{session_id}/clear")
async def clear_conversation_history(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Clear conversation history for session"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        success = await session_service.clear_conversation_history(session_id, user_id, db)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Delete session and all associated data"""
    
    from sqlalchemy import select
    from ...models.session import Session
    
    # Get session
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == user_id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    await db.delete(session)
    await db.commit()

@router.get("/{session_id}/stats", response_model=SessionStats)
async def get_session_stats(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get session statistics"""
    
    from ...services.session_analytics import analytics_service
    
    try:
        stats = await analytics_service.get_session_stats(session_id, user_id, db)
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
```

## 🏪 Marketplace API

### Agent Marketplace Endpoints
```python
# src/api/v1/marketplace.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from ...database.session import get_db
from ...services.marketplace_service import MarketplaceService
from ...schemas.marketplace import (
    MarketplaceEntryCreate, MarketplaceEntryResponse,
    AgentPublishRequest, MarketplaceSearchQuery
)
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/marketplace", tags=["marketplace"])

@router.get("/", response_model=List[MarketplaceEntryResponse])
async def browse_marketplace(
    category: Optional[str] = Query(None, description="Filter by category"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("rating", description="Sort by: rating, downloads, created"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Browse marketplace agents"""
    
    marketplace_service = MarketplaceService()
    
    search_query = MarketplaceSearchQuery(
        category=category,
        tags=tags or [],
        sort_by=sort_by,
        limit=limit,
        offset=offset
    )
    
    entries = await marketplace_service.search_marketplace(search_query, db)
    return entries

@router.post("/publish", response_model=MarketplaceEntryResponse, status_code=status.HTTP_201_CREATED)
async def publish_agent(
    publish_request: AgentPublishRequest,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Publish agent to marketplace"""
    
    marketplace_service = MarketplaceService()
    
    try:
        entry = await marketplace_service.publish_agent(publish_request, user_id, db)
        return entry
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/{entry_id}/download")
async def download_agent(
    entry_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Download/copy agent from marketplace"""
    
    marketplace_service = MarketplaceService()
    
    try:
        agent = await marketplace_service.download_agent(entry_id, user_id, db)
        return {"success": True, "agent_id": agent.id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/categories", response_model=List[str])
async def get_categories(db: AsyncSession = Depends(get_db)):
    """Get available categories"""
    
    marketplace_service = MarketplaceService()
    categories = await marketplace_service.get_categories(db)
    return categories

@router.get("/popular", response_model=List[MarketplaceEntryResponse])
async def get_popular_agents(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Get popular agents"""
    
    from ...services.session_analytics import analytics_service
    
    popular = await analytics_service.get_popular_agents(db, days, limit)
    return popular
```

## 🔧 Authentication & Dependencies

### JWT Authentication
```python
# src/auth/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from uuid import UUID
import os

security = HTTPBearer()

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UUID:
    """Extract user ID from JWT token"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        ALGORITHM = "HS256"
        
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
            
        return UUID(user_id)
        
    except (JWTError, ValueError):
        raise credentials_exception

async def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[UUID]:
    """Extract user ID from JWT token (optional)"""
    
    try:
        return await get_current_user_id(credentials)
    except HTTPException:
        return None
```

## 📊 Error Handling & Validation

### Custom Exception Classes
```python
# src/api/exceptions.py
from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class AgentServiceException(HTTPException):
    """Base exception for agent service errors"""
    
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)

class AgentNotFoundError(AgentServiceException):
    """Agent not found exception"""
    
    def __init__(self, agent_id: str):
        super().__init__(
            detail=f"Agent not found: {agent_id}",
            status_code=status.HTTP_404_NOT_FOUND
        )

class SessionNotFoundError(AgentServiceException):
    """Session not found exception"""
    
    def __init__(self, session_id: str):
        super().__init__(
            detail=f"Session not found: {session_id}",
            status_code=status.HTTP_404_NOT_FOUND
        )

class UnauthorizedAccessError(AgentServiceException):
    """Unauthorized access exception"""
    
    def __init__(self, resource: str):
        super().__init__(
            detail=f"Unauthorized access to {resource}",
            status_code=status.HTTP_403_FORBIDDEN
        )

class ValidationError(AgentServiceException):
    """Validation error exception"""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            detail=f"Validation error in {field}: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
```

## 🔄 Middleware & Request Processing

### Custom Middleware
```python
# src/api/middleware.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {response.status_code} in {duration:.3f}s",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {str(e)} in {duration:.3f}s",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration": duration
                },
                exc_info=True
            )
            
            raise

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 100):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next):
        # Simple IP-based rate limiting
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        minute_ago = current_time - 60
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > minute_ago for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [t for t in self.requests[client_ip] if t > minute_ago]
from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
        
        # Record this request
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)
        
        response = await call_next(request)
        return response
```

## 🚀 WebSocket Support for Real-time Updates

### WebSocket Connection Handler
```python
# src/api/v1/websocket.py
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
from uuid import UUID
import json
import asyncio

from ...auth.websocket_auth import get_websocket_user_id

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[UUID, Set[WebSocket]] = {}
        self.session_connections: Dict[UUID, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: UUID):
        """Connect user WebSocket"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: UUID):
        """Disconnect user WebSocket"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def subscribe_to_session(self, websocket: WebSocket, session_id: UUID):
        """Subscribe WebSocket to session updates"""
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        
        self.session_connections[session_id].add(websocket)
    
    async def broadcast_to_session(self, session_id: UUID, message: dict):
        """Broadcast message to all subscribers of a session"""
        if session_id in self.session_connections:
            disconnected = set()
            
            for websocket in self.session_connections[session_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected.add(websocket)
            
            # Clean up disconnected sockets
            for websocket in disconnected:
                self.session_connections[session_id].discard(websocket)

manager = ConnectionManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: UUID,
    token: str = None
):
    """WebSocket endpoint for real-time updates"""
    
    # Authenticate user (simplified)
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    try:
        # Verify token and get user
        authenticated_user_id = await get_websocket_user_id(token)
        
        if authenticated_user_id != user_id:
            await websocket.close(code=4003, reason="Unauthorized")
            return
        
        await manager.connect(websocket, user_id)
        
        try:
            while True:
                # Handle incoming messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe_session":
                    session_id = UUID(message["session_id"])
                    await manager.subscribe_to_session(websocket, session_id)
                    
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "session_id": str(session_id)
                    }))
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, user_id)
            
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))
```

## 📝 API Response Models

### Standardized Response Format
```python
# src/api/response_models.py
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional, Any, Dict, List
from datetime import datetime

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """Standardized API response format"""
    
    success: bool
    data: Optional[T] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.utcnow()

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response format"""
    
    items: List[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool

class StreamingEventResponse(BaseModel):
    """Streaming event response format"""
    
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.utcnow()
    session_id: Optional[str] = None
    sequence: Optional[int] = None
```

## 🔧 Request Validation

### Custom Validators
```python
# src/api/validators.py
from pydantic import validator
from typing import Any
import re

class ModelConfigValidator:
    """Validators for model configuration"""
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError('max_tokens must be positive')
        return v

class AgentNameValidator:
    """Validators for agent names"""
    
    @validator('name')
    def validate_agent_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Agent name can only contain letters, numbers, hyphens, and underscores')
        return v

class MessageValidator:
    """Validators for messages"""
    
    @validator('content')
    def validate_message_content(cls, v):
        if not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()
```

## 🚀 Key Features

**🌐 Complete REST API:**
- RESTful endpoints following HTTP conventions
- Proper HTTP status codes and error handling
- JWT authentication with role-based access
- Request/response validation with Pydantic

**📡 Real-time Communication:**
- Server-Sent Events (SSE) for streaming agent responses
- WebSocket support for real-time session updates
- Event-driven architecture for live notifications
- Connection management for multiple clients

**🔒 Security & Authentication:**
- JWT token-based authentication
- Role-based access control
- Rate limiting middleware
- Input validation and sanitization

**📊 Monitoring & Observability:**
- Request logging with unique request IDs
- Performance monitoring with response times
- Error tracking and debugging information
- Integration with OpenTelemetry instrumentation

**⚡ Performance Optimization:**
- Connection pooling for database access
- Caching for frequently accessed data
- Pagination for large datasets
- Streaming responses for real-time interaction

**🔧 Developer Experience:**
- Auto-generated OpenAPI documentation
- Interactive API documentation with Swagger UI
- Comprehensive error messages
- Type-safe request/response models

**🔗 Agent Interoperability:**
- Agent2Agent (A2A) Protocol support using official `agent.to_a2a()` method
- FastA2A integration for cross-framework agent communication
- ASGI compatibility for flexible deployment
- Agent exposure management with tracking and lifecycle

## 🌐 Agent2Agent (A2A) Protocol Integration

### A2A Server Deployment
```python
# src/api/routes/a2a.py
from fastapi import APIRouter, Depends
from pydantic_ai import Agent
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from ...core.database import get_db
from ...core.security import get_current_user
from ...services.agent_service import AgentService
from ...models.user import User

router = APIRouter()

@router.post("/{agent_id}/to_a2a")
async def expose_agent_as_a2a(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Expose a PydanticAI agent as an A2A server"""
    
    # Get the agent instance
    agent = await agent_service.get_agent_instance(agent_id, db)
    
    # Convert to A2A server using official method
    a2a_app = agent.to_a2a()
    
    return {
        "message": f"Agent {agent_id} exposed as A2A server",
        "app_type": "ASGI",
        "protocol": "Agent2Agent",
        "usage": "Mount this as a sub-application or run with uvicorn"
    }
```

### FastA2A Service Integration
```python
# src/services/a2a_service.py
from fasta2a import FastA2A
from pydantic_ai import Agent
from typing import Dict, Any
from uuid import UUID

class A2AService:
    """Service for managing Agent2Agent protocol integration"""
    
    def __init__(self):
        self.exposed_agents: Dict[UUID, Any] = {}
    
    def expose_agent_as_a2a(self, agent: Agent, agent_id: UUID) -> Any:
        """Expose a PydanticAI agent as an A2A server using official method"""
        
        # Use official PydanticAI to_a2a method
        a2a_app = agent.to_a2a()
        
        # Store reference for management
        self.exposed_agents[agent_id] = {
            "agent": agent,
            "a2a_app": a2a_app,
            "protocol": "Agent2Agent"
        }
        
        return a2a_app
    
    def get_exposed_agents(self) -> Dict[UUID, Any]:
        """Get list of agents exposed via A2A protocol"""
        return self.exposed_agents
```

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*
            if len(recent_requests) >= self.calls_per_minute: