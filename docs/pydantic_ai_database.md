# 🗄️ Database Schema & Models

> **PostgreSQL database schema optimized for Pydantic AI agents with conversation history**

## 🎯 Overview

Complete database schema designed for multi-tenant AI agent service with persistent conversation history, user management, and agent marketplace functionality.

## 📊 Database Schema Diagram

```mermaid
erDiagram
    User {
        uuid id PK
        string username UK
        string email UK
        string password_hash
        string full_name
        boolean is_active
        datetime created_at
        datetime updated_at
    }
    
    Agent {
        uuid id PK
        string name
        string display_name
        text description
        uuid owner_id FK
        json definition
        json metadata
        boolean is_public
        datetime created_at
        datetime updated_at
    }
    
    Session {
        uuid id PK
        uuid agent_id FK
        uuid user_id FK
        string external_session_id
        json message_history
        json state
        datetime created_at
        datetime updated_at
    }
    
    Event {
        uuid id PK
        uuid session_id FK
        string author
        text content
        json metadata
        boolean partial
        boolean turn_complete
        integer sequence_number
        string error_code
        text error_message
        datetime timestamp
    }
    
    MarketplaceEntry {
        uuid id PK
        uuid agent_id FK
        string category
        json tags
        float rating
        integer download_count
        text changelog
        datetime published_at
    }
    
    User ||--o{ Agent : owns
    User ||--o{ Session : participates
    Agent ||--o{ Session : runs
    Agent ||--o{ MarketplaceEntry : published
    Session ||--o{ Event : contains
```

## 🏗️ Database Models Implementation

### User Management
```python
# src/models/user.py
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime
from passlib.context import CryptContext

from ..database.base import Base

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agents = relationship("Agent", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.password_hash = pwd_context.hash(password)
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.password_hash)
```

### Agent Definition Storage
```python
# src/models/agent.py
from sqlalchemy import Column, String, Text, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

from ..database.base import Base

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Agent configuration stored as JSON
    definition = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Visibility and sharing
    is_public = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="agents")
    sessions = relationship("Session", back_populates="agent", cascade="all, delete-orphan")
    marketplace_entry = relationship("MarketplaceEntry", back_populates="agent", uselist=False)
    
    def __repr__(self):
        return f"<Agent(name='{self.name}', owner='{self.owner_id}')>"
```

### Session & Message History
```python
# src/models/session.py
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

from ..database.base import Base

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PG_UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    external_session_id = Column(String(255), nullable=True, index=True)
    
    # Conversation history stored as JSON using ModelMessagesTypeAdapter
    message_history = Column(JSON, nullable=False, default=list)
    
    # Session state for stateful agents
    state = Column(JSON, nullable=False, default=dict)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Session(agent='{self.agent_id}', user='{self.user_id}')>"
```

### Event Tracking
```python
# src/models/event.py
from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

from ..database.base import Base

class Event(Base):
    __tablename__ = "events"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    
    # Event details
    author = Column(String(50), nullable=False)  # 'user', 'agent', 'system'
    content = Column(Text, nullable=False, default="")
    metadata = Column(JSON, nullable=False, default=dict)
    
    # Streaming support
    partial = Column(Boolean, default=False, nullable=False)
    turn_complete = Column(Boolean, default=True, nullable=False)
    sequence_number = Column(Integer, nullable=False)
    
    # Error handling
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("Session", back_populates="events")
    
    def __repr__(self):
        return f"<Event(author='{self.author}', session='{self.session_id}')>"
```

### Marketplace Entry
```python
# src/models/marketplace.py
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

from ..database.base import Base

class MarketplaceEntry(Base):
    __tablename__ = "marketplace_entries"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PG_UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    
    # Marketplace metadata
    category = Column(String(100), nullable=False, index=True)
    tags = Column(JSON, nullable=False, default=list)
    
    # Analytics
    rating = Column(Float, nullable=True)
    download_count = Column(Integer, default=0, nullable=False)
    
    # Version control
    changelog = Column(Text, nullable=True)
    published_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="marketplace_entry")
    
    def __repr__(self):
        return f"<MarketplaceEntry(agent='{self.agent_id}', category='{self.category}')>"
```

## 🚀 Database Setup & Configuration

### Database Connection
```python
# src/database/connection.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://user:password@localhost:5432/pydantic_ai"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True if os.getenv("SQL_DEBUG") == "true" else False,
    future=True
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

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

### Database Base Class
```python
# src/database/base.py
from sqlalchemy.orm import declarative_base

Base = declarative_base()
```

## 📦 Alembic Migration Setup

### Alembic Configuration
```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.database.base import Base
from src.models import user, agent, session, event, marketplace

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Alembic Config File
```ini
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/pydantic_ai

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

## 🛠️ Database Migration Commands

```bash
# Initialize Alembic (first time only)
uv run alembic init alembic

# Create new migration
uv run alembic revision --autogenerate -m "Add agent and session tables"

# Run migrations
uv run alembic upgrade head

# Downgrade migration
uv run alembic downgrade -1

# Check current revision
uv run alembic current

# Show migration history
uv run alembic history
```

## 📊 Database Indexes for Performance

```sql
-- Additional indexes for performance
CREATE INDEX CONCURRENTLY idx_sessions_user_agent ON sessions(user_id, agent_id);
CREATE INDEX CONCURRENTLY idx_events_session_sequence ON events(session_id, sequence_number);
CREATE INDEX CONCURRENTLY idx_agents_owner_public ON agents(owner_id, is_public);
CREATE INDEX CONCURRENTLY idx_marketplace_category_rating ON marketplace_entries(category, rating DESC);
CREATE INDEX CONCURRENTLY idx_sessions_external_id ON sessions(external_session_id) WHERE external_session_id IS NOT NULL;
```

## 🔧 Database Configuration Options

### Connection Pool Settings
```python
# For production
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,  # 1 hour
)
```

### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/pydantic_ai
SQL_DEBUG=false
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

## 🔄 Key Features

**💾 Message History Storage:**
- JSON storage of Pydantic AI messages using `ModelMessagesTypeAdapter`
- Efficient conversation history retrieval
- Automatic serialization/deserialization

**🔍 Multi-tenant Architecture:**
- User-based agent ownership and access control
- Session isolation between users
- Public/private agent sharing

**📊 Event Tracking:**
- Complete audit trail of all interactions
- Streaming event support with partial messages
- Error tracking and debugging information

**🏪 Marketplace Integration:**
- Agent sharing and discovery
- Rating and download tracking
- Category-based organization

**⚡ Performance Optimized:**
- Strategic database indexes
- Async SQLAlchemy for high concurrency
- Connection pooling for production

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*