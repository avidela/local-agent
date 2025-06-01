"""
Database base configuration and session management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func, text
from sqlalchemy.types import DateTime

from ..config import settings


class Base(DeclarativeBase):
    """Base class for all database models."""
    
    # Common columns for all models
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


# Database engine and session factory
engine = create_async_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session with automatic cleanup."""
    import logging
    logger = logging.getLogger(__name__)
    
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}", exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function for FastAPI to get database session."""
    async with get_db_session() as session:
        yield session


async def init_db() -> None:
    """Initialize database using Alembic migrations"""
    from alembic.config import Config
    from alembic import command
    import asyncio
    import os
    
    print("DEBUG: Starting database initialization...")
    
    # Test database connection first
    print("DEBUG: Testing database connection...")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            print(f"DEBUG: Database connection successful: {result.scalar()}")
    except Exception as e:
        print(f"DEBUG: Database connection failed: {e}")
        raise
    
    def run_migrations():
        """Run Alembic migrations in sync context"""
        print("DEBUG: Starting Alembic migrations...")
        # Get the path to alembic.ini relative to this module
        import pathlib
        base_dir = pathlib.Path(__file__).parent.parent.parent
        alembic_ini_path = base_dir / "alembic.ini"
        migrations_dir = base_dir / "migrations"
        
        print(f"DEBUG: Alembic config path: {alembic_ini_path}")
        print(f"DEBUG: Migrations dir: {migrations_dir}")
        
        alembic_cfg = Config(str(alembic_ini_path))
        # Override the script_location to use absolute path
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        
        # Debug the database URL being used
        db_url = alembic_cfg.get_main_option("sqlalchemy.url")
        print(f"DEBUG: Alembic database URL: {db_url}")
        
        # Override with the correct database URL from settings
        from ..config import settings
        print(f"DEBUG: Settings database URL: {settings.database.url}")
        alembic_cfg.set_main_option("sqlalchemy.url", settings.database.url)
        
        print("DEBUG: Running Alembic upgrade...")
        command.upgrade(alembic_cfg, "head")
        print("DEBUG: Alembic upgrade completed")
    
    # Run migrations in executor to avoid blocking
    print("DEBUG: Running migrations in executor...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_migrations)
    
    print("Database migrations completed successfully")


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()