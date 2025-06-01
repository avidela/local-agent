"""
Database models for PydanticAI Agents Service.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, ForeignKey, String, Text, Boolean, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from uuid import uuid4

from .base import Base


class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    role: Mapped[UserRole] = mapped_column(String(20), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column()
    
    # Relationships
    agents: Mapped[List["Agent"]] = relationship("Agent", back_populates="owner")
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user")


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Agent(Base):
    """Dynamic agent model with tools and configuration."""
    
    __tablename__ = "agents"
    
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Model configuration
    model_provider: Mapped[ModelProvider] = mapped_column(String(20), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Agent configuration
    tools: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=list)  # List of structured tool configurations
    output_type: Mapped[Optional[str]] = mapped_column(String(100))  # Pydantic model name
    retries: Mapped[int] = mapped_column(Integer, default=2)
    
    # Metadata
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    owner: Mapped[User] = relationship("User", back_populates="agents")
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="agent")


class SessionStatus(str, Enum):
    """Session status values."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class Session(Base):
    """Conversation session with persistent message history."""
    
    __tablename__ = "sessions"
    
    session_id: Mapped[str] = mapped_column(
        String(36), 
        default=lambda: str(uuid4()), 
        unique=True, 
        index=True, 
        nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(String(200))
    status: Mapped[SessionStatus] = mapped_column(String(20), default=SessionStatus.ACTIVE)
    
    # Cost tracking
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    request_tokens: Mapped[int] = mapped_column(Integer, default=0)
    response_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped[User] = relationship("User", back_populates="sessions")
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    agent: Mapped[Agent] = relationship("Agent", back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="session")


class MessageRole(str, Enum):
    """Message roles from PydanticAI."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(Base):
    """Individual message in conversation with ModelMessagesTypeAdapter support."""
    
    __tablename__ = "messages"
    
    # Message content - stored as JSONB for ModelMessagesTypeAdapter compatibility
    content: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    role: Mapped[MessageRole] = mapped_column(String(20), nullable=False)
    
    # Multimodal content references
    attachments: Mapped[List[str]] = mapped_column(JSONB, default=list)  # File IDs
    
    # Cost tracking for this message
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Tool calls and responses
    tool_calls: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    tool_response: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Metadata (renamed to avoid SQLAlchemy conflict)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    session: Mapped[Session] = relationship("Session", back_populates="messages")


class FileType(str, Enum):
    """Supported file types for multimodal content."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class File(Base):
    """File storage for multimodal content."""
    
    __tablename__ = "files"
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[FileType] = mapped_column(String(20), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # File processing metadata
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    uploader: Mapped[User] = relationship("User")


class EvaluationStatus(str, Enum):
    """Evaluation status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Evaluation(Base):
    """Evaluation runs for the Pydantic Evals framework."""
    
    __tablename__ = "evaluations"
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[EvaluationStatus] = mapped_column(String(20), default=EvaluationStatus.PENDING)
    
    # Evaluation configuration
    dataset_config: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evaluator_config: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    
    # Results
    results: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    score: Mapped[Optional[float]] = mapped_column(Float)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    passed_cases: Mapped[int] = mapped_column(Integer, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column()
    completed_at: Mapped[Optional[datetime]] = mapped_column()
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    agent: Mapped[Agent] = relationship("Agent")
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    creator: Mapped[User] = relationship("User")


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Workflow(Base):
    """Graph workflows with state persistence."""
    
    __tablename__ = "workflows"
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    workflow_type: Mapped[str] = mapped_column(String(50), nullable=False)  # Graph class name
    status: Mapped[WorkflowStatus] = mapped_column(String(20), default=WorkflowStatus.PENDING)
    
    # Workflow configuration
    graph_config: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    initial_state: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    current_state: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Execution tracking
    current_node: Mapped[Optional[str]] = mapped_column(String(100))
    execution_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=list)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column()
    completed_at: Mapped[Optional[datetime]] = mapped_column()
    paused_at: Mapped[Optional[datetime]] = mapped_column()
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    
    # Relationships
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    creator: Mapped[User] = relationship("User")


class WorkflowExecution(Base):
    """Individual workflow execution steps."""
    
    __tablename__ = "workflow_executions"
    
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str] = mapped_column(String(100), nullable=False)
    node_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    
    # Execution results
    result_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    
    # State snapshots
    state_before: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    state_after: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    workflow_id: Mapped[int] = mapped_column(ForeignKey("workflows.id"), nullable=False)
    workflow: Mapped[Workflow] = relationship("Workflow")