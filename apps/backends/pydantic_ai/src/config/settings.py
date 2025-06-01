"""
Application configuration using Pydantic Settings.
Supports environment variables and .env files.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")
    
    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/pydantic_ai",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=0, description="Max pool overflow")


class ModelProviderSettings(BaseSettings):
    """Model provider API keys and settings."""
    
    model_config = SettingsConfigDict(env_prefix="MODEL_")
    
    # OpenAI settings
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, description="Custom OpenAI base URL")
    
    # Anthropic settings  
    anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # Google Vertex AI settings
    google_application_credentials: Optional[str] = Field(
        default=None, 
        alias="GOOGLE_APPLICATION_CREDENTIALS",
        description="Path to Google service account JSON"
    )
    google_project_id: Optional[str] = Field(default=None, alias="GOOGLE_PROJECT_ID")
    google_location: str = Field(default="us-central1", description="Vertex AI location")
    
    # Default model settings
    default_model: str = Field(default="openai:gpt-4o", description="Default model to use")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: Optional[int] = Field(default=None, ge=1)


class ObservabilitySettings(BaseSettings):
    """Observability and monitoring configuration."""
    
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")
    
    # Logfire settings
    logfire_token: Optional[SecretStr] = Field(default=None, alias="LOGFIRE_TOKEN")
    logfire_service_name: str = Field(default="pydantic-ai-agents", description="Service name")
    logfire_environment: str = Field(default="development", description="Environment name")
    
    # OpenTelemetry settings
    otel_exporter_endpoint: Optional[str] = Field(
        default=None, 
        description="Custom OpenTelemetry exporter endpoint"
    )
    otel_service_name: str = Field(default="pydantic-ai-agents")
    otel_service_version: str = Field(default="0.1.0")
    
    # Tracing settings
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class AuthSettings(BaseSettings):
    """Authentication and authorization settings."""
    
    model_config = SettingsConfigDict(env_prefix="AUTH_")
    
    secret_key: SecretStr = Field(
        default=SecretStr("your-secret-key-change-in-production"),
        description="JWT secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiry in minutes")
    
    # Admin user settings
    admin_username: str = Field(default="admin", description="Default admin username")
    admin_password: SecretStr = Field(
        default=SecretStr("admin"),
        description="Default admin password"
    )


class ApplicationSettings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Basic app settings
    app_name: str = Field(default="PydanticAI Agents Service")
    app_version: str = Field(default="0.1.0")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=True)
    
    # API settings
    api_v1_prefix: str = Field(default="/api/v1")
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    
    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://localhost:8501"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_allow_headers: list[str] = Field(default=["*"])
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    allowed_file_types: list[str] = Field(
        default=["image/jpeg", "image/png", "image/webp", "audio/mpeg", "audio/wav", "video/mp4"],
        description="Allowed MIME types for uploads"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    models: ModelProviderSettings = ModelProviderSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    auth: AuthSettings = AuthSettings()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> ApplicationSettings:
    """Get cached application settings instance."""
    return ApplicationSettings()


# Global settings instance
settings = get_settings()