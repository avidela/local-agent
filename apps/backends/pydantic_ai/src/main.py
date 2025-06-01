"""
Main FastAPI application for PydanticAI Agents Service.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db, close_db
from .observability import setup_observability
from .api import (
    agents_router, sessions_router, auth_router, health_router,
    evaluations_router, workflows_router, tools_router
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    
    # Initialize database
    await init_db()
    print("Database initialized")
    
    yield
    
    # Shutdown
    await close_db()
    print("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Complete PydanticAI agents service with official patterns",
    docs_url=settings.docs_url if not settings.is_production else None,
    redoc_url=settings.redoc_url if not settings.is_production else None,
    lifespan=lifespan,
)

# Setup observability BEFORE adding other middleware or starting the app
setup_observability(app)
print("Observability configured")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include API routers
app.include_router(health_router, tags=["Health"])
app.include_router(auth_router, prefix=f"{settings.api_v1_prefix}/auth", tags=["Authentication"])
app.include_router(agents_router, prefix=f"{settings.api_v1_prefix}/agents", tags=["Agents"])
app.include_router(sessions_router, prefix=f"{settings.api_v1_prefix}/sessions", tags=["Sessions"])
app.include_router(evaluations_router, prefix=f"{settings.api_v1_prefix}/evaluations", tags=["Evaluations"])
app.include_router(workflows_router, prefix=f"{settings.api_v1_prefix}/workflows", tags=["Workflows"])
app.include_router(tools_router, prefix=f"{settings.api_v1_prefix}/tools", tags=["Tools"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": settings.docs_url,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.is_development,
        log_level="info" if settings.is_production else "debug",
    )