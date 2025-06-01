"""
API module for PydanticAI Agents Service.
"""

from .routes.agents import router as agents_router
from .routes.sessions import router as sessions_router
from .routes.auth import router as auth_router
from .routes.health import router as health_router
from .routes.evaluations import router as evaluations_router
from .routes.workflows import router as workflows_router
from .routes.tools import router as tools_router
from .routes.websocket import router as websocket_router

__all__ = [
    "agents_router",
    "sessions_router",
    "auth_router",
    "health_router",
    "evaluations_router",
    "workflows_router",
    "tools_router",
    "websocket_router",
]