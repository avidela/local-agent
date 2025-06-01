"""
Base tool classes for PydanticAI integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    
    status: str = Field(..., description="'success' or 'error'")
    output: Optional[Union[str, List[str], Dict[str, Any]]] = Field(None, description="Tool output on success")
    message: Optional[str] = Field(None, description="Error message on failure")
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information")


class ToolMetadata(BaseModel):
    """Tool metadata for registration and discovery."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="Tool parameter schema")
    examples: List[str] = Field(default_factory=list, description="Usage examples")
    category: str = Field(default="general", description="Tool category")
    enabled: bool = Field(default=True, description="Whether tool is enabled")


class BaseTool(ABC):
    """Base class for all PydanticAI tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tool with configuration."""
        self.config = config or {}
    
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Tool metadata for registration."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """Validate tool parameters against schema."""
        # Basic validation - can be enhanced with pydantic models
        return kwargs
    
    async def __call__(self, **kwargs) -> ToolResult:
        """Make tool callable."""
        try:
            validated_params = self.validate_params(**kwargs)
            return await self.execute(**validated_params)
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Tool execution failed: {str(e)}",
                debug={"error_type": type(e).__name__, "params": kwargs}
            )