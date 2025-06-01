"""
Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ..database.models import ModelProvider, SessionStatus, MessageRole, UserRole, EvaluationStatus, WorkflowStatus


# Base schemas
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime


# User schemas
class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(default=UserRole.USER)


class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase, TimestampMixin):
    """Schema for user responses."""
    id: int
    is_active: bool
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


# Authentication schemas
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """User registration request schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class ResetPasswordRequest(BaseModel):
    """Reset password request schema."""
    username: str
    new_password: str = Field(..., min_length=8, max_length=100)


class UserManagementResponse(BaseModel):
    """User management response schema."""
    message: str
    user_id: int


# Tool and Model configuration schemas
class ToolConfig(BaseModel):
    """Tool configuration schema as documented in pydantic_ai_agents.md"""
    name: str = Field(..., min_length=1, max_length=100, description="Tool name")
    enabled: bool = Field(default=True, description="Whether the tool is enabled")
    plain: bool = Field(default=False, description="Use @agent.tool_plain vs @agent.tool")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific configuration")


class ModelConfig(BaseModel):
    """Model configuration schema as documented in pydantic_ai_agents.md"""
    provider: Literal["google", "anthropic", "openai"] = Field(..., description="Model provider")
    model: str = Field(..., min_length=1, max_length=100, description="Model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens to generate")


# Agent schemas
class AgentBase(BaseModel):
    """Base agent schema with structured tool and model configuration."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: str = Field(..., min_length=1)
    # Structured model configuration
    model_provider: ModelProvider
    model_name: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    # Structured tool configuration - CRITICAL FIX
    tools: List[ToolConfig] = Field(default_factory=list, description="Structured tool configurations")
    output_type: Optional[str] = Field(None, max_length=100)
    retries: int = Field(default=2, ge=0, le=10)
    is_public: bool = Field(default=False)


class AgentCreate(AgentBase):
    """Schema for creating an agent."""
    pass


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, min_length=1)
    model_provider: Optional[ModelProvider] = None
    model_name: Optional[str] = Field(None, min_length=1, max_length=100)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    # Updated to use structured tool configuration
    tools: Optional[List[ToolConfig]] = Field(None, description="Structured tool configurations")
    output_type: Optional[str] = Field(None, max_length=100)
    retries: Optional[int] = Field(None, ge=0, le=10)
    is_public: Optional[bool] = None


class AgentResponse(AgentBase, TimestampMixin):
    """Schema for agent responses."""
    id: int
    is_active: bool
    usage_count: int
    owner_id: int
    
    class Config:
        from_attributes = True


# Session schemas
class SessionBase(BaseModel):
    """Base session schema."""
    title: Optional[str] = Field(None, max_length=200)


class SessionCreate(SessionBase):
    """Schema for creating a session."""
    agent_id: int


class SessionUpdate(BaseModel):
    """Schema for updating a session."""
    title: Optional[str] = Field(None, max_length=200)
    status: Optional[SessionStatus] = None


class SessionResponse(SessionBase, TimestampMixin):
    """Schema for session responses."""
    id: int
    session_id: str
    status: SessionStatus
    total_cost: float
    total_tokens: int
    request_tokens: int
    response_tokens: int
    metadata: Dict[str, Any] = Field(alias="meta_data")
    user_id: int
    agent_id: int
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Message schemas
class MessageCreate(BaseModel):
    """Schema for creating a message."""
    content: Dict[str, Any]
    role: MessageRole
    attachments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Schema for message responses."""
    id: int
    content: Dict[str, Any]
    role: MessageRole
    attachments: List[str]
    cost: float
    tokens: int
    tool_calls: Optional[Dict[str, Any]]
    tool_response: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(alias="meta_data")
    created_at: datetime
    session_id: int
    
    class Config:
        from_attributes = True
        populate_by_name = True


# Agent execution schemas
class AgentRunRequest(BaseModel):
    """Schema for running an agent."""
    prompt: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    message_history: Optional[List[Dict[str, Any]]] = None
    temperature_override: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens_override: Optional[int] = Field(None, gt=0)
    stream: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """Schema for agent run responses."""
    output: Any
    session_id: str
    cost: float
    tokens: int
    request_tokens: int
    response_tokens: int
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Model provider schemas
class ModelInfo(BaseModel):
    """Schema for model information."""
    name: str
    provider: str
    max_tokens: int
    supports_tools: bool
    supports_streaming: bool
    supports_multimodal: bool
    context_window: int


class ProviderStatus(BaseModel):
    """Schema for provider status."""
    provider: ModelProvider
    available: bool
    models: List[ModelInfo]


# File upload schemas
class FileUploadResponse(BaseModel):
    """Schema for file upload responses."""
    file_id: str
    filename: str
    file_type: str
    file_size: int
    processed: bool


# Evaluation schemas
class EvaluationCreate(BaseModel):
    """Schema for creating an evaluation."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    agent_id: int
    dataset_config: Dict[str, Any]
    evaluator_config: Dict[str, Any]
    test_cases: Optional[List[Dict[str, Any]]] = None


class BenchmarkRequest(BaseModel):
    """Schema for benchmark requests."""
    agent_ids: List[int]
    test_cases: List[Dict[str, Any]]
    benchmark_name: Optional[str] = None


class EvaluationUpdate(BaseModel):
    """Schema for updating an evaluation."""
    name: Optional[str] = None
    description: Optional[str] = None
    dataset_config: Optional[Dict[str, Any]] = None
    evaluator_config: Optional[Dict[str, Any]] = None


class EvaluationResponse(BaseModel):
    """Schema for evaluation responses."""
    id: int
    name: str
    description: Optional[str]
    status: EvaluationStatus
    agent_id: int
    dataset_config: Dict[str, Any]
    evaluator_config: Dict[str, Any]
    results: Dict[str, Any]
    score: Optional[float]
    total_cases: int
    passed_cases: int
    failed_cases: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    created_at: datetime
    created_by: int
    
    class Config:
        from_attributes = True


# Workflow schemas
class WorkflowCreate(BaseModel):
    """Schema for creating a workflow."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    workflow_type: str = Field(..., min_length=1, max_length=50)
    graph_config: Dict[str, Any]
    initial_state: Dict[str, Any] = Field(default_factory=dict)


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_type: Optional[str] = None
    graph_config: Optional[Dict[str, Any]] = None
    initial_state: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Schema for workflow responses."""
    id: int
    name: str
    description: Optional[str]
    workflow_type: str
    status: WorkflowStatus
    graph_config: Dict[str, Any]
    initial_state: Dict[str, Any]
    current_state: Dict[str, Any]
    current_node: Optional[str]
    execution_history: List[Dict[str, Any]]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    paused_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    created_at: datetime
    created_by: int
    
    class Config:
        from_attributes = True


# Statistics schemas
class AgentStats(BaseModel):
    """Schema for agent statistics."""
    total_agents: int
    public_agents: int
    private_agents: int
    total_usage: int
    providers: Dict[str, int]


class SessionStats(BaseModel):
    """Schema for session statistics."""
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    total_cost: float
    total_tokens: int
    total_request_tokens: int
    total_response_tokens: int
    average_cost_per_session: float
    average_tokens_per_session: float


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# Health check schema
class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str
    version: str
    timestamp: datetime
    database: str
    providers: List[str]


# Pagination schemas
class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)


class PaginatedResponse(BaseModel):
    """Schema for paginated responses."""
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool