"""
Custom exception types for better error classification and handling.
"""

from typing import Any, Dict, Optional


class PydanticAIException(Exception):
    """Base exception for PydanticAI service."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "PYDANTIC_AI_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(PydanticAIException):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationException(PydanticAIException):
    """Exception for authentication errors."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationException(PydanticAIException):
    """Exception for authorization errors."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR"
        )


class ResourceNotFoundException(PydanticAIException):
    """Exception for resource not found errors."""
    
    def __init__(self, resource_type: str, resource_id: Any):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details={
                "resource_type": resource_type,
                "resource_id": str(resource_id)
            }
        )


class AgentNotFoundException(ResourceNotFoundException):
    """Exception for agent not found errors."""
    
    def __init__(self, agent_id: Any):
        super().__init__("Agent", agent_id)


class SessionNotFoundException(ResourceNotFoundException):
    """Exception for session not found errors."""
    
    def __init__(self, session_id: Any):
        super().__init__("Session", session_id)


class ToolNotFoundException(ResourceNotFoundException):
    """Exception for tool not found errors."""
    
    def __init__(self, tool_name: str):
        super().__init__("Tool", tool_name)


class ModelNotAvailableException(PydanticAIException):
    """Exception for model availability errors."""
    
    def __init__(self, model_name: str, provider: Optional[str] = None):
        message = f"Model '{model_name}' is not available"
        if provider:
            message += f" from provider '{provider}'"
        
        super().__init__(
            message=message,
            error_code="MODEL_NOT_AVAILABLE",
            details={
                "model_name": model_name,
                "provider": provider
            }
        )


class ToolExecutionException(PydanticAIException):
    """Exception for tool execution errors."""
    
    def __init__(self, tool_name: str, error_message: str):
        message = f"Tool '{tool_name}' execution failed: {error_message}"
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details={
                "tool_name": tool_name,
                "error_message": error_message
            }
        )


class ConfigurationException(PydanticAIException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class ExternalServiceException(PydanticAIException):
    """Exception for external service errors."""
    
    def __init__(self, service_name: str, error_message: str, status_code: Optional[int] = None):
        message = f"External service '{service_name}' error: {error_message}"
        details = {"service_name": service_name}
        if status_code:
            details["status_code"] = str(status_code)
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class RateLimitException(PydanticAIException):
    """Exception for rate limiting errors."""
    
    def __init__(self, resource: str, limit: int, window: str):
        message = f"Rate limit exceeded for {resource}: {limit} requests per {window}"
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "resource": resource,
                "limit": limit,
                "window": window
            }
        )


class BusinessLogicException(PydanticAIException):
    """Exception for business logic errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="BUSINESS_LOGIC_ERROR",
            details=details
        )