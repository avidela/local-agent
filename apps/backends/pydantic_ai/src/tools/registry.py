"""
Tool registry for dynamic tool management in PydanticAI agents.
"""

from typing import Dict, List, Optional, Type, Any, Callable
import inspect
from .base import BaseTool, ToolMetadata, ToolResult


class ToolRegistry:
    """Central registry for managing PydanticAI tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._function_tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        name = tool.metadata.name
        self._tools[name] = tool
        self._metadata[name] = tool.metadata
    
    def register_tool_class(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class for lazy instantiation."""
        # Create temporary instance to get metadata
        temp_instance = tool_class()
        name = temp_instance.metadata.name
        self._tool_classes[name] = tool_class
        self._metadata[name] = temp_instance.metadata
    
    def register_function(
        self, 
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "function"
    ) -> None:
        """Register a function as a tool."""
        tool_name = name or func.__name__
        
        # Extract parameter schema from function signature
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_info = {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "required": param.default == inspect.Parameter.empty
            }
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            parameters[param_name] = param_info
        
        # Create metadata
        metadata = ToolMetadata(
            name=tool_name,
            description=description or func.__doc__ or f"Function: {tool_name}",
            parameters=parameters,
            category=category
        )
        
        self._function_tools[tool_name] = func
        self._metadata[tool_name] = metadata
    
    def get_tool(self, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseTool]:
        """Get a tool instance by name."""
        if name in self._tools:
            return self._tools[name]
        elif name in self._tool_classes:
            # Lazy instantiation
            tool_instance = self._tool_classes[name](config)
            self._tools[name] = tool_instance
            return tool_instance
        return None
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a function tool by name."""
        return self._function_tools.get(name)
    
    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """List available tool names."""
        tools = []
        for name, metadata in self._metadata.items():
            if category and metadata.category != category:
                continue
            if enabled_only and not metadata.enabled:
                continue
            tools.append(name)
        return tools
    
    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        return self._metadata.get(name)
    
    def get_all_metadata(self, category: Optional[str] = None) -> Dict[str, ToolMetadata]:
        """Get all tool metadata."""
        if category:
            return {
                name: metadata 
                for name, metadata in self._metadata.items() 
                if metadata.category == category
            }
        return self._metadata.copy()
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        removed = False
        if name in self._tools:
            del self._tools[name]
            removed = True
        if name in self._tool_classes:
            del self._tool_classes[name]
            removed = True
        if name in self._function_tools:
            del self._function_tools[name]
            removed = True
        if name in self._metadata:
            del self._metadata[name]
            removed = True
        return removed
    
    async def execute_tool(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        # Try tool instance first
        tool = self.get_tool(name, config)
        if tool:
            return await tool(**kwargs)
        
        # Try function tool
        func = self.get_function(name)
        if func:
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)
                
                return ToolResult(
                    status="success",
                    output=result,
                    message=None,
                    debug=None
                )
            except Exception as e:
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Function execution failed: {str(e)}",
                    debug={"error_type": type(e).__name__, "params": kwargs}
                )
        
        return ToolResult(
            status="error",
            output=None,
            message=f"Tool '{name}' not found",
            debug={"available_tools": list(self._metadata.keys())}
        )


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry