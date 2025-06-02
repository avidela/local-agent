"""
Database module for PydanticAI Agents Service.
"""

from .base import Base, get_db, get_db_session, init_db, close_db, engine, async_session_factory
from .models import (
    User, UserRole,
    Agent, ModelProvider,
    Session, SessionStatus,
    Message, MessageRole,
    Evaluation, EvaluationStatus,
    Workflow, WorkflowStatus, WorkflowExecution,
)

__all__ = [
    # Base database functionality
    "Base",
    "get_db",
    "get_db_session", 
    "init_db",
    "close_db",
    "engine",
    "async_session_factory",
    
    # User models
    "User",
    "UserRole",
    
    # Agent models
    "Agent",
    "ModelProvider",
    
    # Session and message models
    "Session",
    "SessionStatus",
    "Message", 
    "MessageRole",
    
    
    # Evaluation models
    "Evaluation",
    "EvaluationStatus",
    
    # Workflow models
    "Workflow",
    "WorkflowStatus",
    "WorkflowExecution",
]