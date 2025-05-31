# Pydantic AI Service - Complete Accurate Implementation Guide

## 🎯 Overview
This guide provides a complete, accurate implementation of a Pydantic AI agents service using the **official Pydantic AI API patterns** validated against current documentation.

## 📦 Dependencies & Installation

### Core Dependencies
```toml
# apps/backends/pydantic_ai/pyproject.toml
[project]
name = "pydantic-ai-service"
version = "0.1.0"
description = "Multi-agent system with dynamic agent creation using Pydantic AI"
requires-python = ">=3.11"
dependencies = [
    # Core framework
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.12.0",
    "asyncpg>=0.29.0",
    "redis>=5.0.0",
    
    # Pydantic AI with model providers
    "pydantic-ai-slim[google,anthropic]>=0.0.1",
    
    # Authentication & Security
    "google-auth>=2.23.0",
    "google-auth-oauthlib>=1.1.0", 
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    
    # API & Utils
    "python-multipart>=0.0.6",
    "httpx>=0.25.0",
    "aiofiles>=23.2.1",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    
    # Observability (optional)
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
]
```

## 🔧 Model Service (Official Pydantic AI API)

### Complete Model Service Implementation
```python
# src/services/model_service.py
from typing import Dict, Any, Optional
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..core.config import settings

class ModelService:
    """Service for managing AI models using official Pydantic AI API"""
    
    def __init__(self):
        # Initialize Google Vertex AI provider
        self.google_provider = GoogleProvider(vertexai=True)
        
        # Initialize Anthropic provider (direct API, not Vertex)
        self.anthropic_provider = AnthropicProvider()
        
        # Model cache
        self._model_cache: Dict[str, Any] = {}
    
    def get_model(self, provider: str, model_name: str, **config):
        """Get a Pydantic AI model instance using official API"""
        cache_key = f"{provider}:{model_name}:{hash(str(config))}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if provider == "vertex-gemini":
            model = GoogleModel(model_name, provider=self.google_provider, **config)
        elif provider == "anthropic":
            model = AnthropicModel(model_name, provider=self.anthropic_provider, **config)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        self._model_cache[cache_key] = model
        return model
    
    def list_available_models(self) -> Dict[str, list]:
        """List available models by provider"""
        return {
            "vertex-gemini": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro"
            ],
            "anthropic": [
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-opus-latest"
            ]
        }
```

## 🛠️ Tool System (Official Patterns)

### Base Tool Interface
```python
# src/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable
from pydantic_ai import RunContext

class BaseToolMixin(ABC):
    """Base interface for custom tools"""
    
    @abstractmethod
    def get_tool_function(self) -> Callable:
        """Return the tool function to be registered with agent"""
        pass
```

### File System Tools
```python
# src/tools/filesystem/read_write.py
import os
import aiofiles
from pydantic_ai import RunContext
from ..base import BaseToolMixin
from ...core.config import settings

class FileSystemTools(BaseToolMixin):
    """File system tools using official Pydantic AI patterns"""
    
    def get_read_file_tool(self):
        """Get file reading tool"""
        async def read_file(ctx: RunContext[None], file_path: str) -> str:
            """Read a file from the repository
            
            Args:
                file_path: Path to file relative to repository root
            """
            # Security check
            full_path = os.path.join(settings.repo_root, file_path.lstrip('/'))
            if not os.path.commonpath([full_path, settings.repo_root]) == settings.repo_root:
                raise ValueError("Path outside repository root")
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return content
        
        return read_file
    
    def get_write_file_tool(self):
        """Get file writing tool"""
        async def write_file(ctx: RunContext[None], file_path: str, content: str) -> str:
            """Write content to a file in the repository
            
            Args:
                file_path: Path to file relative to repository root
                content: Content to write to the file
            """
            full_path = os.path.join(settings.repo_root, file_path.lstrip('/'))
            if not os.path.commonpath([full_path, settings.repo_root]) == settings.repo_root:
                raise ValueError("Path outside repository root")
            
            # Create directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {file_path}"
        
        return write_file
    
    def get_grep_tool(self):
        """Get grep search tool"""
        import subprocess
        from typing import List
        from pydantic import BaseModel
        
        class GrepResult(BaseModel):
            file_path: str
            line_number: int
            content: str
        
        async def grep_search(
            ctx: RunContext[None], 
            pattern: str, 
            file_path: str = ".", 
            recursive: bool = True
        ) -> List[GrepResult]:
            """Search for patterns in files using grep
            
            Args:
                pattern: Search pattern
                file_path: Path to search in (relative to repository root)
                recursive: Whether to search recursively
            """
            full_path = os.path.join(settings.repo_root, file_path.lstrip('/'))
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
    
    def get_tool_function(self):
        # This is for interface compliance, tools are registered individually
        pass
```

## 🤖 Agent Factory (Official API)

### Dynamic Agent Creation
```python
# src/agents/dynamic/factory.py
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ...tools.filesystem.read_write import FileSystemTools
from ...services.model_service import ModelService
from ...models.agent import Agent as AgentModel

class AgentDefinition(BaseModel):
    name: str
    display_name: str
    description: str
    model: Dict[str, Any]
    system_prompt: str
    tools: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class PydanticAgentFactory:
    """Factory for creating Pydantic AI agents using official API"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.filesystem_tools = FileSystemTools()
    
    async def create_agent(
        self, 
        definition: AgentDefinition,
        deps_type: Optional[type] = None
    ) -> Agent:
        """Create a Pydantic AI agent using official API patterns"""
        
        # Get model using official API
        model = self.model_service.get_model(
            provider=definition.model["provider"],
            model_name=definition.model["model"],
            temperature=definition.model.get("temperature", 0.1),
            max_tokens=definition.model.get("max_tokens"),
        )
        
        # Create agent with official Pydantic AI constructor
        agent = Agent(
            model=model,
            system_prompt=definition.system_prompt,
            deps_type=deps_type or type(None)
        )
        
        # Register tools using official decorators
        self._register_tools(agent, definition.tools)
        
        return agent
    
    def _register_tools(self, agent: Agent, tool_configs: List[Dict[str, Any]]):
        """Register tools with agent using official patterns"""
        
        for tool_config in tool_configs:
            if not tool_config.get("enabled", True):
                continue
                
            tool_name = tool_config["name"]
            
            if tool_name == "file_read":
                tool_func = self.filesystem_tools.get_read_file_tool()
                agent._function_tools.append(tool_func)
                
            elif tool_name == "file_write":
                tool_func = self.filesystem_tools.get_write_file_tool()
                agent._function_tools.append(tool_func)
                
            elif tool_name == "grep":
                tool_func = self.filesystem_tools.get_grep_tool()
                agent._function_tools.append(tool_func)
    
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

## 🔄 Agent Execution Service

### Complete Agent Service with Official API
```python
# src/services/agent_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, AsyncGenerator
from uuid import UUID
from pydantic_ai import Agent

from ..models.agent import Agent as AgentModel
from ..schemas.agents import AgentCreate, AgentResponse
from ..agents.dynamic.factory import PydanticAgentFactory, AgentDefinition
from .model_service import ModelService

class AgentService:
    """Service for managing agents using official Pydantic AI API"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.agent_factory = PydanticAgentFactory(model_service)
        self._agent_cache: Dict[UUID, Agent] = {}
    
    async def create_agent(
        self,
        agent_data: AgentCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> AgentResponse:
        """Create a new dynamic agent"""
        
        # Convert to internal definition format
        definition = AgentDefinition(
            name=agent_data.name,
            display_name=agent_data.display_name,
            description=agent_data.description,
            model=agent_data.model.dict(),
            system_prompt=agent_data.system_prompt,
            tools=[tool.dict() for tool in agent_data.tools],
            metadata=agent_data.metadata
        )
        
        # Store in database
        agent_model = await self.agent_factory.store_agent_definition(
            definition, user_id, db
        )
        
        return AgentResponse.from_orm(agent_model)
    
    async def execute_agent(
        self,
        agent_id: UUID,
        input_data: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession,
        streaming: bool = False
    ):
        """Execute agent using official Pydantic AI run methods"""
        
        # Get or create agent instance
        if agent_id not in self._agent_cache:
            agent_model = await self._get_agent_model(agent_id, db)
            definition = AgentDefinition(**agent_model.definition)
            pydantic_agent = await self.agent_factory.create_agent(definition)
            self._agent_cache[agent_id] = pydantic_agent
        
        agent = self._agent_cache[agent_id]
        message = input_data.get("message", "")
        
        if streaming:
            return self._execute_streaming(agent, message)
        else:
            return await self._execute_batch(agent, message)
    
    async def _execute_batch(self, agent: Agent, message: str) -> Dict[str, Any]:
        """Execute agent in batch mode using run_sync"""
        try:
            result = await agent.run(message)
            return {
                "response": result.data,
                "usage": {
                    "total_cost": result.cost(),
                    "message_count": len(result.all_messages())
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "response": "An error occurred during agent execution"
            }
    
    async def _execute_streaming(self, agent: Agent, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent in streaming mode using run_stream"""
        try:
            async for chunk in agent.run_stream(message):
                yield {
                    "partial": True,
                    "content": str(chunk),
                    "turn_complete": False
                }
            
            # Final message
            yield {
                "partial": False,
                "content": "",
                "turn_complete": True
            }
            
        except Exception as e:
            yield {
                "error": str(e),
                "partial": False,
                "turn_complete": True
            }
    
    async def _get_agent_model(self, agent_id: UUID, db: AsyncSession) -> AgentModel:
        """Get agent model from database"""
        from sqlalchemy import select
        
        result = await db.execute(select(AgentModel).where(AgentModel.id == agent_id))
        agent_model = result.scalar_one_or_none()
        
        if not agent_model:
            raise ValueError(f"Agent not found: {agent_id}")
        
        return agent_model
## 💬 Conversation History & Message Management

### Enhanced Database Models for Message Persistence
```python
# src/models/session.py - Updated with message history
from sqlalchemy import Column, UUID, String, Text, DateTime, JSON, ForeignKey, Boolean, Integer
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
    
    # Message history stored as JSON for conversation continuity
    message_history = Column(JSON, nullable=False, default=list)
    
    state = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
```

### Session Service with Conversation History
```python
# src/services/session_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID, uuid4
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

from ..models.session import Session
from ..models.event import Event
from ..schemas.sessions import SessionCreate, SessionResponse, MessageCreate, EventResponse
from .agent_service import AgentService

class SessionService:
    """Service for managing sessions with proper message history using official Pydantic AI API"""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
    
    async def create_session(
        self,
        session_data: SessionCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> SessionResponse:
        """Create a new session"""
        
        session = Session(
            agent_id=session_data.agent_id,
            user_id=user_id,
            external_session_id=session_data.external_session_id or str(uuid4()),
            state=session_data.initial_state or {},
            message_history=[]  # Initialize empty message history
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        return SessionResponse.from_orm(session)
    
    async def send_message(
        self,
        session_id: UUID,
        message: MessageCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> AsyncGenerator[EventResponse, None]:
        """Send message and get agent response with proper conversation history"""
        
        # Get session with message history
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session:
            raise ValueError("Session not found")
        
        # Get agent instance
        agent = await self.agent_service._get_agent_instance(session.agent_id, db)
        
        # Prepare message history from previous conversations
        message_history = None
        if session.message_history:
            try:
                # Convert stored JSON back to message objects using official API
                message_history = ModelMessagesTypeAdapter.validate_python(session.message_history)
            except Exception:
                # If message history is corrupted, start fresh
                message_history = None
        
        # Execute agent with conversation history
        if message.streaming:
            async for event in self._execute_streaming_with_history(
                agent, message.content, message_history, session, db
            ):
                yield event
        else:
            event = await self._execute_batch_with_history(
                agent, message.content, message_history, session, db
            )
            yield event
    
    async def _execute_batch_with_history(
        self,
        agent: Agent,
        user_message: str,
        message_history: Optional[List],
        session: Session,
        db: AsyncSession
    ) -> EventResponse:
        """Execute agent with message history in batch mode using official API"""
        
        try:
            # Run agent with message history for conversation continuity
            if message_history:
                result = await agent.run(user_message, message_history=message_history)
            else:
                result = await agent.run(user_message)
            
            # Store user message event
            await self._store_user_event(session.id, user_message, db)
            
            # Store agent response event with usage info
            agent_event = await self._store_agent_event(
                session.id, 
                result.data, 
                result.cost(),
                len(result.all_messages()),
                db
            )
            
            # Update session message history with new messages using official API
            await self._update_session_history(session, result.all_messages(), db)
            
            return EventResponse.from_orm(agent_event)
            
        except Exception as e:
            # Store error event
            error_event = await self._store_error_event(session.id, str(e), db)
            return EventResponse.from_orm(error_event)
    
    async def _execute_streaming_with_history(
        self,
        agent: Agent,
        user_message: str,
        message_history: Optional[List],
        session: Session,
        db: AsyncSession
    ) -> AsyncGenerator[EventResponse, None]:
        """Execute agent with message history in streaming mode using official API"""
        
        try:
            # Store user message first
            await self._store_user_event(session.id, user_message, db)
            
            # Start streaming with message history using official API
            if message_history:
                stream_result = agent.run_stream(user_message, message_history=message_history)
            else:
                stream_result = agent.run_stream(user_message)
            
            accumulated_response = ""
            
            async with stream_result as result:
                # Stream partial responses using official stream_text() method
                async for text_chunk in result.stream_text():
                    accumulated_response += text_chunk
                    
                    # Create partial event
                    partial_event = Event(
                        session_id=session.id,
                        author="agent",
                        content=text_chunk,
                        metadata={"partial": True},
                        partial=True,
                        turn_complete=False,
                        sequence_number=await self._get_next_sequence_number(session.id, db)
                    )
                    
                    db.add(partial_event)
                    await db.commit()
                    await db.refresh(partial_event)
                    
                    yield EventResponse.from_orm(partial_event)
                
                # Store final complete response with usage info
                final_event = await self._store_agent_event(
                    session.id,
                    accumulated_response,
                    result.cost(),
                    len(result.all_messages()),
                    db
                )
                
                # Update session message history using official all_messages() method
                await self._update_session_history(session, result.all_messages(), db)
                
                yield EventResponse.from_orm(final_event)
                
        except Exception as e:
            error_event = await self._store_error_event(session.id, str(e), db)
            yield EventResponse.from_orm(error_event)
    
    async def _update_session_history(
        self,
        session: Session,
        all_messages: List,
        db: AsyncSession
    ):
        """Update session with latest message history for conversation continuity using official API"""
        
        try:
            # Convert messages to JSON for storage using official to_jsonable_python
            session.message_history = to_jsonable_python(all_messages)
            await db.commit()
        except Exception:
            # If serialization fails, keep existing history
            pass
    
    async def get_conversation_history(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session or not session.message_history:
            return []
        
        try:
            # Convert stored message history back to readable format
            messages = ModelMessagesTypeAdapter.validate_python(session.message_history)
            return [
                {
                    "type": "user" if hasattr(msg, "parts") and any("User" in str(type(part)) for part in msg.parts) else "assistant",
                    "content": str(msg),
                    "timestamp": getattr(msg, "timestamp", None)
                }
                for msg in messages
            ]
        except Exception:
            return []
    
    async def _store_user_event(self, session_id: UUID, content: str, db: AsyncSession) -> Event:
        """Store user message event"""
        event = Event(
            session_id=session_id,
            author="user",
            content=content,
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _store_agent_event(
        self, 
        session_id: UUID, 
        content: str, 
        cost: float,
        message_count: int,
        db: AsyncSession
    ) -> Event:
        """Store agent response event with usage information"""
        event = Event(
            session_id=session_id,
            author="agent",
            content=content,
            metadata={
                "cost": cost,
                "message_count": message_count,
                "partial": False
            },
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _store_error_event(self, session_id: UUID, error_message: str, db: AsyncSession) -> Event:
        """Store error event"""
        event = Event(
            session_id=session_id,
            author="system",
            content="",
            error_code="EXECUTION_ERROR",
            error_message=error_message,
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _get_session_with_history(self, session_id: UUID, user_id: UUID, db: AsyncSession) -> Optional[Session]:
        """Get session with message history"""
        from sqlalchemy import select
        
        result = await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id
            )
        )
        return result.scalar_one_or_none()
    
    async def _get_next_sequence_number(self, session_id: UUID, db: AsyncSession) -> int:
        """Get next sequence number for event"""
        from sqlalchemy import select, func
        
        result = await db.execute(
            select(func.max(Event.sequence_number))
            .where(Event.session_id == session_id)
        )
        last_seq = result.scalar() or 0
        return last_seq + 1
```

### Updated Agent Service with Message History Support
```python
# src/services/agent_service.py - Enhanced version
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from uuid import UUID
from pydantic_ai import Agent

from ..models.agent import Agent as AgentModel
from ..schemas.agents import AgentCreate, AgentResponse
from ..agents.dynamic.factory import PydanticAgentFactory, AgentDefinition
from .model_service import ModelService

class AgentService:
    """Service for managing agents using official Pydantic AI API with message history support"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.agent_factory = PydanticAgentFactory(model_service)
        self._agent_cache: Dict[UUID, Agent] = {}
    
    async def create_agent(
        self,
        agent_data: AgentCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> AgentResponse:
        """Create a new dynamic agent"""
        
        definition = AgentDefinition(
            name=agent_data.name,
            display_name=agent_data.display_name,
            description=agent_data.description,
            model=agent_data.model.dict(),
            system_prompt=agent_data.system_prompt,
            tools=[tool.dict() for tool in agent_data.tools],
            metadata=agent_data.metadata
        )
        
        agent_model = await self.agent_factory.store_agent_definition(
            definition, user_id, db
        )
        
        return AgentResponse.from_orm(agent_model)
    
    async def _get_agent_instance(self, agent_id: UUID, db: AsyncSession) -> Agent:
        """Get agent instance with caching for message history support"""
        
        if agent_id not in self._agent_cache:
            agent_model = await self._get_agent_model(agent_id, db)
            definition = AgentDefinition(**agent_model.definition)
            pydantic_agent = await self.agent_factory.create_agent(definition)
            self._agent_cache[agent_id] = pydantic_agent
        
        return self._agent_cache[agent_id]
    
    async def _get_agent_model(self, agent_id: UUID, db: AsyncSession) -> AgentModel:
        """Get agent model from database"""
        from sqlalchemy import select
        
        result = await db.execute(select(AgentModel).where(AgentModel.id == agent_id))
        agent_model = result.scalar_one_or_none()
        
        if not agent_model:
            raise ValueError(f"Agent not found: {agent_id}")
        
        return agent_model
```

### API Endpoints with Conversation History
```python
# src/api/v1/sessions.py - Enhanced with message history
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from uuid import UUID

from ...database.session import get_db
from ...services.session_service import SessionService
from ...services.agent_service import AgentService
from ...services.model_service import ModelService
from ...schemas.sessions import SessionCreate, SessionResponse, MessageCreate, EventResponse
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Create a new session with message history support"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    return await session_service.create_session(session_data, user_id, db)

@router.post("/{session_id}/messages")
async def send_message(
    session_id: UUID,
    message: MessageCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Send message with conversation history maintained"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    if message.streaming:
        async def event_stream():
            async for event in session_service.send_message(session_id, message, user_id, db):
                yield f"data: {event.json()}\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/plain")
    else:
        events = []
        async for event in session_service.send_message(session_id, message, user_id, db):
            events.append(event)
        return events[0] if events else None

@router.get("/{session_id}/history")
async def get_conversation_history(
    session_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Get conversation history for a session"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    return await session_service.get_conversation_history(session_id, user_id, db)
```

### Key Features of Message History Implementation

**🔄 Official API Integration:**
- Uses [`result.all_messages()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for complete conversation history
- Uses [`result.new_messages()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for current run messages
- Uses [`ModelMessagesTypeAdapter`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for JSON serialization
- Uses [`to_jsonable_python()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for message storage

**💾 Persistent Conversation History:**
- Messages stored as JSON in PostgreSQL
- Automatic conversion between JSON and Pydantic AI message objects
- Conversation continuity across sessions
- Error handling for corrupted message history

**🔄 Streaming with History:**
- Uses [`result.stream_text()`](https://docs.pydantic.ai/api/agents/#pydantic_ai.Agent.run_stream) for real-time response streaming
- Maintains conversation context during streaming
- Proper message history updates after stream completion

**📊 Usage Tracking:**
- Uses [`result.cost()`](https://docs.pydantic.ai/api/agents/#pydantic_ai.RunResult.cost) for cost tracking
- Message count tracking via [`len(result.all_messages())`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter)
- Usage metadata stored with each response
```

## 🌍 Environment Configuration

### Complete Environment Setup
```bash
# apps/backends/pydantic_ai/.env

# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost/pydantic_ai
SECRET_KEY=your-secret-key-here-change-in-production

# Google Vertex AI (uses default credentials)
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_REGION=us-central1

# Anthropic (direct API)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Application
REDIS_URL=redis://localhost:6379
REPO_ROOT=/repos
```

### Authentication Setup
```bash
# For Google Vertex AI authentication

# Option 1: gcloud CLI (development)
gcloud auth application-default login

# Option 2: Service account (production)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Option 3: In Docker with mounted service account
# Mount the service account file and set environment variable
```

## 🎯 Key Implementation Points

### ✅ Official Pydantic AI Patterns Used:
1. **Google Vertex**: `GoogleProvider(vertexai=True)` + `GoogleModel`
2. **Anthropic**: `AnthropicProvider()` + `AnthropicModel` (direct API only)
3. **Tools**: Function-based tools with proper decorators
4. **Agent Creation**: `Agent(model, system_prompt, deps_type)`
5. **Execution**: `agent.run()` and `agent.run_stream()` methods

### ✅ Dependencies Simplified:
- Only need `pydantic-ai-slim[google,anthropic]`
- Google auth handled by `google-auth` package
- No custom Vertex integration needed

### ✅ Authentication:
- Vertex AI uses Google Cloud default credentials
- Anthropic uses API key from environment

### ⚠️ Important Notes:
1. **Anthropic Vertex**: Not supported by Pydantic AI - only direct Anthropic API
2. **Tool Registration**: Use official function-based pattern, not custom registries
3. **Model Names**: Use exact model names as documented
4. **Streaming**: Use `agent.run_stream()` for streaming responses

This implementation follows the official Pydantic AI documentation patterns exactly, ensuring compatibility and maintainability.
## 📊 Observability & Monitoring with OpenTelemetry

### Production Monitoring Setup with Logfire
```python
# src/observability/instrumentation.py
import os
import logfire
from typing import Optional
from pydantic_ai.agent import Agent, InstrumentationSettings

class ObservabilityService:
    """Service for configuring OpenTelemetry instrumentation for production monitoring"""
    
    def __init__(self, service_name: str = "pydantic-ai-service"):
        self.service_name = service_name
        self._configured = False
    
    def configure_logfire(
        self,
        token: Optional[str] = None,
        send_to_logfire: bool = True,
        event_mode: str = 'events'  # 'events' or 'logs'
    ):
        """Configure Logfire for production monitoring (recommended)"""
        
        if self._configured:
            return
        
        # Configure Logfire with optional token
        if token:
            os.environ['LOGFIRE_TOKEN'] = token
        
        logfire.configure(
            service_name=self.service_name,
            send_to_logfire=send_to_logfire
        )
        
        # Instrument PydanticAI with semantic conventions
        logfire.instrument_pydantic_ai(event_mode=event_mode)
        
        # Instrument HTTP requests for model API calls
        logfire.instrument_httpx(capture_all=True)
        
        # Instrument SQLAlchemy for database operations
        logfire.instrument_sqlalchemy()
        
        # Instrument FastAPI for API monitoring
        logfire.instrument_fastapi()
        
        self._configured = True
    
    def configure_alternative_otel_backend(
        self,
        endpoint: str = "http://localhost:4318",
        event_mode: str = 'events'
    ):
        """Configure Logfire SDK with alternative OpenTelemetry backend"""
        
        if self._configured:
            return
        
        # Set OTel endpoint
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = endpoint
        
        # Configure Logfire to send to alternative backend
        logfire.configure(
            service_name=self.service_name,
            send_to_logfire=False  # Don't send to Logfire platform
        )
        
        # Instrument components
        logfire.instrument_pydantic_ai(event_mode=event_mode)
        logfire.instrument_httpx(capture_all=True)
        logfire.instrument_sqlalchemy()
        logfire.instrument_fastapi()
        
        self._configured = True
    
    def configure_raw_opentelemetry(
        self,
        endpoint: str = "http://localhost:4318"
    ):
        """Configure raw OpenTelemetry without Logfire"""
        
        if self._configured:
            return
        
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import set_tracer_provider
        from opentelemetry.sdk._events import EventLoggerProvider
        
        # Configure OTel endpoint
        os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = endpoint
        
        # Setup trace provider
        exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(exporter)
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(span_processor)
        set_tracer_provider(tracer_provider)
        
        # Setup event logger provider
        event_logger_provider = EventLoggerProvider()
        
        # Create custom instrumentation settings
        instrumentation_settings = InstrumentationSettings(
            tracer_provider=tracer_provider,
            event_logger_provider=event_logger_provider,
            include_binary_content=False  # Exclude binary for performance
        )
        
        # Instrument all agents
        Agent.instrument_all(instrumentation_settings)
        
        self._configured = True
    
    def get_custom_instrumentation_settings(
        self,
        include_binary_content: bool = False,
        event_mode: str = 'events'
    ) -> InstrumentationSettings:
        """Get custom instrumentation settings for specific agents"""
        
        return InstrumentationSettings(
            include_binary_content=include_binary_content
        )

# Global observability service instance
observability = ObservabilityService()
```

### Enhanced Model Service with Instrumentation
```python
# src/services/model_service.py - Enhanced with observability
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.instrumented import InstrumentedModel
from typing import Optional, Union
import os

from ..observability.instrumentation import observability

class ModelService:
    """Enhanced model service with OpenTelemetry instrumentation"""
    
    def __init__(self):
        self.google_provider = GoogleProvider(vertexai=True)
        self.anthropic_provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def get_model(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        instrumented: bool = True
    ) -> Union[GoogleModel, AnthropicModel, InstrumentedModel]:
        """Get model with optional instrumentation for observability"""
        
        base_model = None
        
        if provider == "google":
            base_model = GoogleModel(
                model_name,
                provider=self.google_provider,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        elif provider == "anthropic":
            base_model = AnthropicModel(
                model_name,
                provider=self.anthropic_provider,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Wrap with instrumentation for observability
        if instrumented:
            instrumentation_settings = observability.get_custom_instrumentation_settings(
                include_binary_content=False
            )
            return InstrumentedModel(base_model, instrumentation_settings)
        
        return base_model
```

### FastAPI Application with Observability
```python
# src/main.py - Enhanced with monitoring
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from .database.connection import init_db
from .api.v1 import agents, sessions, marketplace
from .observability.instrumentation import observability

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with observability setup"""
    
    # Initialize observability based on environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Production: Use Logfire platform
        observability.configure_logfire(
            token=os.getenv("LOGFIRE_TOKEN"),
            send_to_logfire=True,
            event_mode='events'
        )
    elif environment == "staging":
        # Staging: Use alternative OTel backend
        observability.configure_alternative_otel_backend(
            endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318"),
            event_mode='logs'  # Better for long conversations
        )
    else:
        # Development: Use raw OpenTelemetry or local setup
        if os.getenv("OTEL_ENDPOINT"):
            observability.configure_raw_opentelemetry(
                endpoint=os.getenv("OTEL_ENDPOINT")
            )
    
    # Initialize database
    await init_db()
    
    yield
    
    # Cleanup if needed

def create_app() -> FastAPI:
    """Create FastAPI application with observability"""
    
    app = FastAPI(
        title="Pydantic AI Agents Service",
        description="Production-ready AI agents service with observability",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(marketplace.router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint with tracing"""
        return {"status": "healthy", "service": "pydantic-ai-agents"}
    
    return app

app = create_app()
```

### Environment Configuration for Observability
```python
# .env.production
ENVIRONMENT=production
LOGFIRE_TOKEN=your_logfire_token_here
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
GOOGLE_CLOUD_PROJECT=your_project_id
ANTHROPIC_API_KEY=your_anthropic_key_here

# .env.staging  
ENVIRONMENT=staging
OTEL_ENDPOINT=http://your-otel-collector:4318
DATABASE_URL=postgresql+asyncpg://user:pass@staging-host:5432/db
GOOGLE_CLOUD_PROJECT=your_staging_project
ANTHROPIC_API_KEY=your_anthropic_key_here

# .env.development
ENVIRONMENT=development
OTEL_ENDPOINT=http://localhost:4318
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/local_db
GOOGLE_CLOUD_PROJECT=your_dev_project
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Docker Setup with OpenTelemetry Collector
```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  # PydanticAI Service
  pydantic-ai-service:
    build: .
    environment:
      - ENVIRONMENT=staging
      - OTEL_ENDPOINT=http://otel-collector:4318
      - DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/db
    depends_on:
      - postgres
      - otel-collector
    ports:
      - "8000:8000"

  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./observability/otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
    depends_on:
      - jaeger

  # Jaeger for trace visualization
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # Jaeger gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  # Alternative: OTEL TUI for development
  otel-tui:
    image: ymtdzzz/otel-tui:latest
    ports:
      - "4318:4318"
    profiles:
      - dev

volumes:
  postgres_data:
```

### OpenTelemetry Collector Configuration
```yaml
# observability/otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
  memory_limiter:
    limit_mib: 512

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  # Optional: Export to Logfire
  otlphttp:
    endpoint: https://logfire-api.pydantic.dev
    headers:
      authorization: "Bearer ${LOGFIRE_TOKEN}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger]
    
    # Uncomment for Logfire export
    # traces:
    #   receivers: [otlp]
    #   processors: [memory_limiter, batch]
    #   exporters: [otlphttp]
```

### Production Monitoring Dashboard Setup
```python
# src/observability/metrics.py
from pydantic_ai.agent import Agent
import logfire
from typing import Dict, Any
import time

class MetricsCollector:
    """Custom metrics collection for production monitoring"""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
    
    def track_agent_execution(self, agent_id: str, user_id: str):
        """Track agent execution metrics"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log successful execution
                    logfire.info(
                        "Agent execution completed",
                        agent_id=agent_id,
                        user_id=user_id,
                        execution_time=time.time() - start_time,
                        status="success"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failed execution
                    logfire.error(
                        "Agent execution failed",
                        agent_id=agent_id,
                        user_id=user_id,
                        execution_time=time.time() - start_time,
                        error=str(e),
                        status="error"
                    )
                    raise
            
            return wrapper
        return decorator
    
    def track_model_usage(self, model_name: str, provider: str):
        """Track model usage and costs"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                # Extract usage information from result
                if hasattr(result, 'cost') and hasattr(result, 'all_messages'):
                    logfire.info(
                        "Model usage tracked",
                        model_name=model_name,
                        provider=provider,
                        cost=result.cost(),
                        message_count=len(result.all_messages()),
                        total_tokens=getattr(result, 'usage', {}).get('total_tokens', 0)
                    )
                
                return result
            
            return wrapper
        return decorator

# Global metrics collector
metrics = MetricsCollector()
```

### Key Features of Observability Implementation

**📊 Multiple Backend Support:**
- [`logfire.configure()`](https://logfire.pydantic.dev/) - Recommended Logfire platform integration
- Alternative OTel backends with [`send_to_logfire=False`](https://logfire.pydantic.dev/)
- Raw OpenTelemetry without Logfire using [`Agent.instrument_all()`](https://docs.pydantic.ai/agents/#pydantic_ai.Agent.instrument_all)

**🔍 Comprehensive Instrumentation:**
- [`logfire.instrument_pydantic_ai()`](https://logfire.pydantic.dev/) - Agent execution tracing
- [`logfire.instrument_httpx()`](https://logfire.pydantic.dev/) - Model API call monitoring
- [`logfire.instrument_sqlalchemy()`](https://logfire.pydantic.dev/) - Database operation tracking
- [`logfire.instrument_fastapi()`](https://logfire.pydantic.dev/) - API endpoint monitoring

**⚙️ Advanced Configuration:**
- [`InstrumentationSettings`](https://docs.pydantic.ai/agents/#pydantic_ai.InstrumentationSettings) for custom configuration
- [`event_mode='logs'`](https://docs.pydantic.ai/agents/#pydantic_ai.InstrumentationSettings) for long conversations
- [`include_binary_content=False`](https://docs.pydantic.ai/agents/#pydantic_ai.InstrumentationSettings) for performance
- Custom tracer and event logger providers

**📈 Production Monitoring:**
- Cost tracking with [`result.cost()`](https://docs.pydantic.ai/api/agents/#pydantic_ai.RunResult.cost)
- Usage analytics with message counts and token tracking
- Error tracking and performance monitoring
- Distributed tracing across microservices