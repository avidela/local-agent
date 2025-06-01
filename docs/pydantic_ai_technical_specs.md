# Pydantic AI Technical Specifications

## 📋 Implementation Requirements

This document provides detailed technical specifications for implementing the Pydantic AI agents service. Each section includes exact code patterns, configuration details, and step-by-step instructions.

## 🗄️ Database Models - Exact Implementation

### Base Model
```python
# src/models/base.py
from sqlalchemy import Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

### User Model
```python
# src/models/user.py
from sqlalchemy import Column, String, Boolean, Text, Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .base import BaseModel
import enum

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"

class User(BaseModel):
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    profile = Column(JSONB, default={})
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    owned_agents = relationship("Agent", back_populates="owner")
    sessions = relationship("Session", back_populates="user")
    agent_ratings = relationship("AgentRating", back_populates="user")
    granted_permissions = relationship("AgentPermission", back_populates="user")
```

### Agent Model
```python
# src/models/agent.py
from sqlalchemy import Column, String, Text, Boolean, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from .base import BaseModel
import enum

class AgentScope(str, enum.Enum):
    USER = "user"           # Private to user
    SYSTEM = "system"       # System-wide
    MARKETPLACE = "marketplace"  # Published to marketplace

class AgentType(str, enum.Enum):
    STATIC = "static"       # Code-defined agent
    DYNAMIC = "dynamic"     # User-created agent
    TEMPLATE = "template"   # Template for creating agents

class Agent(BaseModel):
    __tablename__ = "agents"
    
    name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    scope = Column(Enum(AgentScope), default=AgentScope.USER)
    agent_type = Column(Enum(AgentType), default=AgentType.DYNAMIC)
    definition = Column(JSONB, nullable=False)  # Complete agent definition
    metadata = Column(JSONB, default={})
    version = Column(String(20), default="1.0.0")
    is_active = Column(Boolean, default=True)
    is_template = Column(Boolean, default=False)
    
    # Relationships
    owner = relationship("User", back_populates="owned_agents")
    versions = relationship("AgentVersion", back_populates="agent")
    permissions = relationship("AgentPermission", back_populates="agent")
    marketplace_entry = relationship("AgentMarketplace", back_populates="agent", uselist=False)
    sessions = relationship("Session", back_populates="agent")
```

### Complete Database Schema
```python
# src/models/__init__.py
from .base import BaseModel
from .user import User, UserRole
from .agent import Agent, AgentScope, AgentType
from .session import Session
from .event import Event
from .execution import AgentExecution, ExecutionStatus, WorkflowType
from .marketplace import AgentMarketplace, AgentRating
from .observability import ObservabilityTrace
from .artifact import Artifact

__all__ = [
    "BaseModel", "User", "UserRole", "Agent", "AgentScope", "AgentType",
    "Session", "Event", "AgentExecution", "ExecutionStatus", "WorkflowType",
    "AgentMarketplace", "AgentRating", "ObservabilityTrace", "Artifact"
]
```

## 🔧 Configuration Management

### Environment Configuration
```python
# src/core/config.py
from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_echo: bool = Field(False, env="DATABASE_ECHO")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Authentication
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Model Providers (Vertex AI)
    google_cloud_project: str = Field(..., env="GOOGLE_CLOUD_PROJECT")
    google_cloud_region: str = Field("us-central1", env="GOOGLE_CLOUD_REGION")
    
    # Observability
    otel_service_name: str = Field("pydantic-ai-service", env="OTEL_SERVICE_NAME")
    otel_exporter_endpoint: Optional[str] = Field(None, env="OTEL_EXPORTER_ENDPOINT")
    
    # File System
    repo_root: str = Field("/repos", env="REPO_ROOT")
    upload_dir: str = Field("./uploads", env="UPLOAD_DIR")
    
    # CORS
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## 🤖 Agent System Implementation

### Agent Factory
```python
# src/agents/dynamic/factory.py
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from typing import Dict, Any, List
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from ..base.agent import BaseAgentDefinition
from ...tools.registry import ToolRegistry
from ...services.model_service import ModelService
from ...models.agent import Agent as AgentModel

class AgentDefinition(BaseAgentDefinition):
    name: str
    display_name: str
    description: str
    model: Dict[str, Any]
    system_prompt: str
    tools: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class AgentFactory:
    def __init__(self, tool_registry: ToolRegistry, model_service: ModelService):
        self.tool_registry = tool_registry
        self.model_service = model_service
    
    async def create_agent(
        self, 
        definition: AgentDefinition,
        user_id: UUID,
        db: AsyncSession
    ) -> Agent:
        """Create a Pydantic AI agent from definition"""
        
        # Get model
        model = self.model_service.get_model(
            provider=definition.model["provider"],
            model_name=definition.model["model"],
            **definition.model.get("config", {})
        )
        
        # Get tools
        tools = []
        for tool_def in definition.tools:
            if tool_def["enabled"]:
                tool = self.tool_registry.get_tool(tool_def["name"])
                configured_tool = tool.configure(tool_def.get("config", {}))
                tools.append(configured_tool)
        
        # Create agent
        agent = Agent(
            model=model,
            system_prompt=definition.system_prompt,
            tools=tools,
            deps_type=type(None)  # Will be enhanced based on needs
        )
        
        return agent
    
    async def store_agent_definition(
        self,
        definition: AgentDefinition,
        user_id: UUID,
        db: AsyncSession
    ) -> AgentModel:
        """Store agent definition in database"""
        
        agent_model = AgentModel(
            name=definition.name,
            display_name=definition.display_name,
            description=definition.description,
            owner_id=user_id,
            definition=definition.dict(),
            metadata=definition.metadata
        )
        
        db.add(agent_model)
        await db.commit()
        await db.refresh(agent_model)
        
        return agent_model
```

### Tool Registry
```python
# src/tools/registry.py
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> Callable:
        pass

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def register_tool(self, name: str, tool: BaseTool):
        self._tools[name] = tool
    
    def get_tool(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")
        return self._tools[name]
    
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
    
    def _register_default_tools(self):
        from .filesystem.read_write import FileReadTool, FileWriteTool
        from .filesystem.search import GrepTool, FindTool
        from .search.web_search import WebSearchTool
        
        self.register_tool("file_read", FileReadTool())
        self.register_tool("file_write", FileWriteTool())
        self.register_tool("grep", GrepTool())
        self.register_tool("find", FindTool())
        self.register_tool("web_search", WebSearchTool())
```

## 🌐 API Implementation

### FastAPI Application Setup
```python
# src/api/main.py
from fastapi import FastAPI, Middleware
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..core.config import settings
from ..core.database import init_db
from ..core.observability import setup_observability
from .routes import auth, users, agents, sessions, marketplace, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    setup_observability()
    yield
    # Shutdown
    # Add cleanup code here

def create_app() -> FastAPI:
    app = FastAPI(
        title="Pydantic AI Agents Service",
        description="Multi-agent system with dynamic agent creation",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Routes
    app.include_router(auth.router, prefix="/auth", tags=["authentication"])
    app.include_router(users.router, prefix="/users", tags=["users"])
    app.include_router(agents.router, prefix="/agents", tags=["agents"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(marketplace.router, prefix="/marketplace", tags=["marketplace"])
    app.include_router(health.router, prefix="/health", tags=["health"])
    
    return app
```

### Agent Routes
```python
# src/api/routes/agents.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from ...core.database import get_db
from ...core.security import get_current_user
from ...models.user import User
from ...models.agent import Agent
from ...schemas.agents import AgentCreate, AgentResponse, AgentUpdate
from ...services.agent_service import AgentService

router = APIRouter()

@router.post("/", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Create a new dynamic agent"""
    return await agent_service.create_agent(agent_data, current_user.id, db)

@router.get("/", response_model=List[AgentResponse])
async def list_user_agents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """List current user's agents"""
    return await agent_service.list_user_agents(current_user.id, db)

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Get agent details"""
    agent = await agent_service.get_agent(agent_id, db)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check permissions
    if agent.owner_id != current_user.id and agent.scope != "system":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: UUID,
    agent_data: AgentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Update agent"""
    return await agent_service.update_agent(agent_id, agent_data, current_user.id, db)

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Delete agent"""
    await agent_service.delete_agent(agent_id, current_user.id, db)
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/execute")
async def execute_agent(
    agent_id: UUID,
    input_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_service: AgentService = Depends()
):
    """Execute agent with input"""
    return await agent_service.execute_agent(agent_id, input_data, current_user.id, db)
```

## 📊 Database Configuration

### Database Setup
```python
# src/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from .config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    poolclass=NullPool,  # For async
    future=True
)

# Create session maker
async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        yield session

async def init_db() -> None:
    """Initialize database using Alembic migrations"""
    from alembic.config import Config
    from alembic import command
    import asyncio
    
    def run_migrations():
        """Run Alembic migrations in sync context"""
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    
    # Run migrations in executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_migrations)
    
    print("Database migrations completed successfully")
```

### Alembic Configuration
```python
# migrations/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from src.models.base import BaseModel
from src.core.config import settings

# Import all models
from src.models import *

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = BaseModel.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## 🔐 Authentication Implementation

### JWT Authentication
```python
# src/core/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .config import settings
from .database import get_db
from ..models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    return user
```

## 📦 UV Workspace Configuration

### pyproject.toml
```toml
# apps/backends/pydantic_ai/pyproject.toml
[project]
name = "pydantic-ai-service"
version = "0.1.0"
description = "Multi-agent system with dynamic agent creation using Pydantic AI"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.12.0",
    "asyncpg>=0.29.0",
    "redis>=5.0.0",
    "pydantic-ai>=0.0.1",
    "google-cloud-aiplatform>=1.38.0",
    "anthropic[vertex]>=0.7.0",
    "google-auth>=2.23.0",
    "google-auth-oauthlib>=1.1.0",
    "google-auth-httplib2>=0.2.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-multipart>=0.0.6",
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
    "opentelemetry-instrumentation-fastapi>=0.41b0",
    "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "httpx>=0.25.0",
    "aiofiles>=23.2.1",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.9.1",
    "isort>=5.12.0",
    "mypy>=1.6.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

## 🐳 Docker Configuration

### Dockerfile
```dockerfile
# apps/backends/pydantic_ai/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Step-by-Step Implementation Guide

### Phase 1: Foundation Setup
1. **Create UV workspace structure**
2. **Implement database models with exact field definitions**
3. **Set up FastAPI application with proper middleware**
4. **Implement authentication system with JWT**
5. **Create basic CRUD operations for users and agents**
6. **Set up Alembic migrations**
7. **Add OpenTelemetry observability**

### Phase 2: Agent System
1. **Implement tool registry with default tools**
2. **Create agent factory for dynamic agent creation**
3. **Implement agent service with validation**
4. **Add agent execution with proper error handling**
5. **Implement session management**
6. **Add event tracking and storage**

### Phase 3: Marketplace
1. **Implement marketplace models and schemas**
2. **Create marketplace service with search functionality**
3. **Add rating system**
4. **Implement agent publishing workflow**
5. **Add marketplace API endpoints**

This technical specification provides all the concrete implementation details needed for a less powerful model to successfully implement the Pydantic AI service.