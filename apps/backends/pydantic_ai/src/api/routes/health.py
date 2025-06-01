"""
Health check API routes.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...database import get_db
from ...services import model_provider_service
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns:
        Application health status
    """
    
    # Check database connection
    try:
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Get available providers
    providers = [p.value for p in model_provider_service.get_supported_providers()]
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "unhealthy",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        database=db_status,
        providers=providers,
    )


@router.get("/health/database")
async def database_health(db: AsyncSession = Depends(get_db)):
    """
    Check database health.
    
    Returns:
        Database connection status
    """
    
    try:
        await db.execute("SELECT 1")
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow()}


@router.get("/health/providers")
async def providers_health():
    """
    Check model providers health.
    
    Returns:
        Status of all configured model providers
    """
    
    providers_status = []
    
    for provider in model_provider_service.get_supported_providers():
        models = model_provider_service.get_models_by_provider(provider)
        
        providers_status.append({
            "provider": provider.value,
            "available": len(models) > 0,
            "model_count": len(models),
            "models": list(models.keys()),
        })
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "providers": providers_status,
    }