"""
Global exception handling middleware for consistent error responses and logging.
"""

import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Union, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...config import settings
from ..exceptions import (
    PydanticAIException, ValidationException, AuthenticationException,
    AuthorizationException, ResourceNotFoundException, ModelNotAvailableException,
    ToolExecutionException, ConfigurationException, ExternalServiceException,
    RateLimitException, BusinessLogicException
)

# Configure logger
logger = logging.getLogger(__name__)


class ErrorResponse:
    """Structured error response format."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int,
        request_id: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.request_id = request_id
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        response = {
            "error": True,
            "message": self.message,
            "error_code": self.error_code,
            "request_id": self.request_id,
            "timestamp": self.timestamp
        }
        
        if self.details:
            response["details"] = self.details
        
        return response


def get_request_id(request: Request) -> str:
    """Get or generate request ID for tracking."""
    # Check if request ID exists in headers
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        # Generate new request ID
        request_id = str(uuid.uuid4())
    
    return request_id


def log_exception(
    request: Request,
    exception: Exception,
    request_id: str,
    status_code: int
) -> None:
    """Log exception with request context."""
    
    log_data = {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "status_code": status_code,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "user_agent": request.headers.get("User-Agent"),
        "client_ip": request.client.host if request.client else None,
    }
    
    # Add traceback for 5xx errors
    if status_code >= 500:
        log_data["traceback"] = traceback.format_exc()
    
    # Log at appropriate level
    if status_code >= 500:
        logger.error("Server error occurred", extra=log_data)
    elif status_code >= 400:
        logger.warning("Client error occurred", extra=log_data)
    else:
        logger.info("Request processed with error", extra=log_data)


async def pydantic_ai_exception_handler(request: Request, exc: PydanticAIException) -> JSONResponse:
    """Handle custom PydanticAI exceptions."""
    request_id = get_request_id(request)
    
    # Map exception types to HTTP status codes
    status_code_map = {
        ValidationException: status.HTTP_400_BAD_REQUEST,
        AuthenticationException: status.HTTP_401_UNAUTHORIZED,
        AuthorizationException: status.HTTP_403_FORBIDDEN,
        ResourceNotFoundException: status.HTTP_404_NOT_FOUND,
        ModelNotAvailableException: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ToolExecutionException: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ConfigurationException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ExternalServiceException: status.HTTP_502_BAD_GATEWAY,
        RateLimitException: status.HTTP_429_TOO_MANY_REQUESTS,
        BusinessLogicException: status.HTTP_422_UNPROCESSABLE_ENTITY,
    }
    
    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    error_response = ErrorResponse(
        message=exc.message,
        error_code=exc.error_code,
        status_code=status_code,
        request_id=request_id,
        details=exc.details
    )
    
    log_exception(request, exc, request_id, status_code)
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.to_dict(),
        headers={"X-Request-ID": request_id}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPExceptions."""
    request_id = get_request_id(request)
    
    error_response = ErrorResponse(
        message=exc.detail,
        error_code="HTTP_ERROR",
        status_code=exc.status_code,
        request_id=request_id
    )
    
    log_exception(request, exc, request_id, exc.status_code)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.to_dict(),
        headers={"X-Request-ID": request_id}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = get_request_id(request)
    
    # Extract validation error details
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "value": error.get("input")
        })
    
    error_response = ErrorResponse(
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        request_id=request_id,
        details={"validation_errors": validation_errors}
    )
    
    log_exception(request, exc, request_id, status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.to_dict(),
        headers={"X-Request-ID": request_id}
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle SQLAlchemy database errors."""
    request_id = get_request_id(request)
    
    # Don't expose database details in production
    if settings.is_production:
        message = "Database operation failed"
        details = {}
    else:
        message = f"Database error: {str(exc)}"
        details = {"database_error": str(exc)}
    
    error_response = ErrorResponse(
        message=message,
        error_code="DATABASE_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
        details=details
    )
    
    log_exception(request, exc, request_id, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.to_dict(),
        headers={"X-Request-ID": request_id}
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exceptions."""
    request_id = get_request_id(request)
    
    # Don't expose internal details in production
    if settings.is_production:
        message = "Internal server error"
        details = {}
    else:
        message = f"Unhandled error: {str(exc)}"
        details = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
    
    error_response = ErrorResponse(
        message=message,
        error_code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=request_id,
        details=details
    )
    
    log_exception(request, exc, request_id, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.to_dict(),
        headers={"X-Request-ID": request_id}
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers for the FastAPI application."""
    
    # Custom PydanticAI exceptions
    app.add_exception_handler(PydanticAIException, pydantic_ai_exception_handler)  # type: ignore
    
    # FastAPI built-in exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore
    app.add_exception_handler(ValidationError, validation_exception_handler)  # type: ignore
    
    # Database errors
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)  # type: ignore
    
    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore
    
    logger.info("Global exception handlers configured")


# Middleware for adding request ID to all responses
async def request_id_middleware(request: Request, call_next):
    """Middleware to add request ID to response headers."""
    request_id = get_request_id(request)
    
    # Add request ID to request state for use in handlers
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response