# Pydantic AI Implementation Steps

## 📋 Phase 1: Foundation Setup (Days 1-3)

### Step 1: Create Project Structure
```bash
# From the project root directory
mkdir -p apps/backends/pydantic_ai
cd apps/backends/pydantic_ai

# Create all required directories
mkdir -p src/api/routes
mkdir -p src/agents/{base,static,dynamic,workflows}
mkdir -p src/tools/{filesystem,search,multimodal,custom}
mkdir -p src/models
mkdir -p src/schemas
mkdir -p src/services
mkdir -p src/core
mkdir -p src/utils
mkdir -p migrations/versions
mkdir -p templates
mkdir -p tests
mkdir -p uploads
```

### Step 2: Create pyproject.toml
```bash
# Copy the exact content from pydantic_ai_technical_specs.md
# File: apps/backends/pydantic_ai/pyproject.toml
```

### Step 3: Create Base Configuration Files
Create these files in exact order:

**File 1: `.env` (create this file)**
```bash
# apps/backends/pydantic_ai/.env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/pydantic_ai
SECRET_KEY=your-secret-key-here-change-in-production
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_REGION=us-central1
REDIS_URL=redis://localhost:6379
REPO_ROOT=/repos
```

**File 2: `src/core/config.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - Configuration Management section
```

**File 3: `src/models/base.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - Base Model section
```

### Step 4: Create Database Models (Create in this exact order)
1. `src/models/user.py` - Copy from technical specs
2. `src/models/agent.py` - Copy from technical specs  
3. `src/models/session.py` - Create this file:
```python
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from .base import BaseModel

class Session(BaseModel):
    __tablename__ = "sessions"
    
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    external_session_id = Column(String(255), nullable=True)
    state = Column(JSONB, default={})
    configuration = Column(JSONB, default={})
    
    # Relationships
    agent = relationship("Agent", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    events = relationship("Event", back_populates="session", order_by="Event.sequence_number")
    executions = relationship("AgentExecution", back_populates="session")
    artifacts = relationship("Artifact", back_populates="session")
```

4. `src/models/event.py` - Create this file:
```python
from sqlalchemy import Column, String, Text, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from .base import BaseModel

class Event(BaseModel):
    __tablename__ = "events"
    
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    agent_execution_id = Column(UUID(as_uuid=True), ForeignKey("agent_executions.id"), nullable=True)
    author = Column(String(50), nullable=False)  # 'user', 'agent', 'system'
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, default={})
    actions = Column(JSONB, default={})
    partial = Column(Boolean, default=False)
    turn_complete = Column(Boolean, default=True)
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    sequence_number = Column(Integer, nullable=False)
    
    # Relationships
    session = relationship("Session", back_populates="events")
    execution = relationship("AgentExecution", back_populates="events")
```

5. Continue with remaining models from technical specs...

### Step 5: Create Database Connection
**File: `src/core/database.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - Database Setup section
```

### Step 6: Setup Alembic
```bash
# Initialize Alembic
alembic init migrations

# Replace migrations/env.py with code from technical specs
```

**File: `alembic.ini`**
```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = 

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

## 📋 Phase 2: Core API Setup (Days 4-6)

### Step 7: Create Pydantic Schemas
Create these files with exact code from `pydantic_ai_code_templates.md`:

1. `src/schemas/agents.py`
2. `src/schemas/sessions.py` 
3. `src/schemas/marketplace.py`
4. `src/schemas/auth.py` - Create this file:
```python
from pydantic import BaseModel, EmailStr, Field

class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True
```

### Step 8: Create Authentication System
**File: `src/core/security.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - JWT Authentication section
```

### Step 9: Create API Routes
Create these route files in order:

1. `src/api/routes/health.py`:
```python
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "pydantic-ai-service"
    }
```

2. `src/api/routes/auth.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import timedelta

from ...core.database import get_db
from ...core.security import verify_password, create_access_token, get_password_hash
from ...models.user import User
from ...schemas.auth import UserLogin, UserRegister, Token, UserResponse
from ...core.config import settings

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    # Get user
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    # Check if user exists
    result = await db.execute(
        select(User).where(
            (User.username == user_data.username) | (User.email == user_data.email)
        )
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse.from_orm(user)
```

3. `src/api/routes/agents.py` - Copy from code templates
4. Continue with other routes...

### Step 10: Create FastAPI Application
**File: `src/api/main.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - FastAPI Application Setup
```

**File: `main.py` (in root of pydantic_ai directory)**
```python
import uvicorn
from src.api.main import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

## 📋 Phase 3: Agent System (Days 7-10)

### Step 11: Create Tool Registry
**File: `src/tools/base.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

class BaseTool(ABC):
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> Callable:
        """Configure and return the tool function"""
        pass
```

**File: `src/tools/registry.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - Tool Registry section
```

### Step 12: Create File System Tools
**File: `src/tools/filesystem/read_write.py`**
```python
# Copy exact code from pydantic_ai_code_templates.md - File System Tools section
```

**File: `src/tools/filesystem/search.py`**
```python
import os
import subprocess
from typing import Dict, Any, Callable, List
from pydantic_ai import tool
from pydantic import BaseModel

from ..base import BaseTool
from ...core.config import settings

class GrepResult(BaseModel):
    file_path: str
    line_number: int
    content: str

class GrepTool(BaseTool):
    def configure(self, config: Dict[str, Any]) -> Callable:
        @tool
        async def grep_search(pattern: str, file_path: str = ".", recursive: bool = True) -> List[GrepResult]:
            """Search for patterns in files using grep"""
            
            full_path = os.path.join(settings.repo_root, file_path.lstrip('/'))
            
            # Security check
            if not os.path.commonpath([full_path, settings.repo_root]) == settings.repo_root:
                raise ValueError("Path outside repository root")
            
            cmd = ["grep", "-n"]
            if recursive:
                cmd.append("-r")
            cmd.extend([pattern, full_path])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=settings.repo_root)
                
                results = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append(GrepResult(
                                file_path=parts[0],
                                line_number=int(parts[1]),
                                content=parts[2]
                            ))
                
                return results
                
            except Exception as e:
                raise ValueError(f"Grep search failed: {str(e)}")
        
        return grep_search
```

### Step 13: Create Agent Factory
**File: `src/agents/dynamic/factory.py`**
```python
# Copy exact code from pydantic_ai_technical_specs.md - Agent Factory section
```

### Step 14: Create Services
**File: `src/services/agent_service.py`**
```python
# Copy exact code from pydantic_ai_code_templates.md - Agent Service section
```

**File: `src/services/session_service.py`**
```python
# Copy exact code from pydantic_ai_code_templates.md - Session Service section
```

## 📋 Phase 4: Testing & Deployment (Days 11-14)

### Step 15: Create Database Migration
```bash
# Create first migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

### Step 16: Create Tests
**File: `tests/conftest.py`**
```python
# Copy exact code from pydantic_ai_code_templates.md - Test Configuration section
```

**File: `tests/test_agents.py`**
```python
# Copy exact code from pydantic_ai_code_templates.md - Agent Tests section
```

### Step 17: Create Docker Setup
**File: `Dockerfile`**
```dockerfile
# Copy exact code from pydantic_ai_technical_specs.md - Docker Configuration section
```

**File: `docker-compose-pydantic-ai.yml` (in project root)**
```yaml
version: '3.8'

services:
  pydantic-ai:
    build:
      context: ./apps/backends/pydantic_ai
      dockerfile: Dockerfile
    ports:
      - "8002:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/pydantic_ai
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=your-secret-key-change-in-production
    depends_on:
      - postgres
      - redis
    volumes:
      - ${HOME}/Solutions:/repos
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=pydantic_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## 🚀 Running the Service

### Step 18: Start Development Environment
```bash
# Install dependencies
cd apps/backends/pydantic_ai
uv sync

# Start database
docker-compose -f ../../../docker-compose-pydantic-ai.yml up -d postgres redis

# Run migrations
uv run alembic upgrade head

# Start the service
uv run uvicorn main:app --reload --port 8002
```

### Step 19: Test the API
```bash
# Test health endpoint
curl http://localhost:8002/health

# Register a user
curl -X POST http://localhost:8002/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "testpass123"}'

# Login
curl -X POST http://localhost:8002/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass123"
```

## 📝 Implementation Checklist

### Phase 1 - Foundation ✅
- [ ] Project structure created
- [ ] pyproject.toml configured  
- [ ] Database models implemented
- [ ] Alembic migrations setup
- [ ] Basic API structure created

### Phase 2 - Core API ✅
- [ ] Authentication system working
- [ ] User registration/login endpoints
- [ ] Agent CRUD endpoints
- [ ] Session management endpoints
- [ ] Error handling implemented

### Phase 3 - Agent System ✅  
- [ ] Tool registry functional
- [ ] File system tools working
- [ ] Agent factory creating dynamic agents
- [ ] Agent execution working
- [ ] Session message handling

### Phase 4 - Production Ready ✅
- [ ] Database migrations applied
- [ ] Tests passing
- [ ] Docker containers running
- [ ] API documentation accessible
- [ ] Observability configured

## 🔧 Troubleshooting Common Issues

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose -f docker-compose-pydantic-ai.yml ps

# Check logs
docker-compose -f docker-compose-pydantic-ai.yml logs postgres
```

### Import Errors
```bash
# Make sure you're in the right directory
cd apps/backends/pydantic_ai

# Install dependencies
uv sync

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

### Migration Issues
```bash
# Reset migrations (CAREFUL - this deletes data)
uv run alembic downgrade base
uv run alembic upgrade head

# Or create new migration
uv run alembic revision --autogenerate -m "Fix schema"
```

This step-by-step guide provides the exact sequence and commands needed to implement the complete Pydantic AI service successfully.