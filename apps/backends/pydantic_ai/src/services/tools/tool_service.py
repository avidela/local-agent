"""
Tool management service for PydanticAI agents.
"""

from typing import Dict, List, Optional, Any, Callable
from ...tools.registry import get_tool_registry, ToolRegistry
from ...tools.base import BaseTool, ToolMetadata, ToolResult
from ...tools.filesystem import (
    ReadFileTool, WriteFileTool, ListDirectoryTool, DeleteFileTool,
    CopyFileTool, MoveFileTool, FileSearchTool, GrepFileTool,
    FindFilesTool, DiffFilesTool
)
from ...tools.builtin import CalculatorTool, WebSearchTool
from ...observability import trace_tool_execution, add_span_attributes, record_exception


class ToolService:
    """Service for managing and executing tools."""
    
    def __init__(self):
        """Initialize the tool service."""
        self.registry = get_tool_registry()
        self._initialize_builtin_tools()
    
    def _initialize_builtin_tools(self):
        """Register all built-in tools."""
        # Filesystem tools
        self.registry.register_tool_class(ReadFileTool)
        self.registry.register_tool_class(WriteFileTool)
        self.registry.register_tool_class(ListDirectoryTool)
        self.registry.register_tool_class(DeleteFileTool)
        self.registry.register_tool_class(CopyFileTool)
        self.registry.register_tool_class(MoveFileTool)
        self.registry.register_tool_class(FileSearchTool)
        self.registry.register_tool_class(GrepFileTool)
        self.registry.register_tool_class(FindFilesTool)
        self.registry.register_tool_class(DiffFilesTool)
        
        # Built-in tools
        self.registry.register_tool_class(CalculatorTool)
        self.registry.register_tool_class(WebSearchTool)
        
        # Register some simple function tools
        self._register_function_tools()
    
    def _register_function_tools(self):
        """Register simple function-based tools."""
        
        def echo(message: str) -> str:
            """Echo a message back."""
            return f"Echo: {message}"
        
        def current_time() -> str:
            """Get the current time."""
            from datetime import datetime
            return datetime.now().isoformat()
        
        def generate_uuid() -> str:
            """Generate a new UUID."""
            import uuid
            return str(uuid.uuid4())
        
        # Register functions
        self.registry.register_function(
            echo, 
            description="Echo a message back",
            category="utility"
        )
        self.registry.register_function(
            current_time,
            description="Get the current time in ISO format", 
            category="utility"
        )
        self.registry.register_function(
            generate_uuid,
            description="Generate a new UUID",
            category="utility"
        )
    
    def get_available_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """Get list of available tool names."""
        return self.registry.list_tools(category=category, enabled_only=enabled_only)
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        return self.registry.get_metadata(tool_name)
    
    def get_all_tool_metadata(self, category: Optional[str] = None) -> Dict[str, ToolMetadata]:
        """Get metadata for all tools."""
        return self.registry.get_all_metadata(category=category)
    
    async def execute_tool(
        self,
        tool_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute a tool by name with observability tracing."""
        
        # Extract operation name from kwargs or use generic
        operation = kwargs.get('operation', 'execute')
        
        # Filter out 'operation' from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if isinstance(v, (str, int, float, bool)) and k != 'operation'}
        
        with trace_tool_execution(
            tool_name=tool_name,
            operation=operation,
            config_keys=list(config.keys()) if config else [],
            **filtered_kwargs
        ):
            try:
                # Filter out 'operation' from kwargs for tool execution as well
                tool_kwargs = {k: v for k, v in kwargs.items() if k != 'operation'}
                result = await self.registry.execute_tool(tool_name, config=config, **tool_kwargs)
                
                # Add success metrics
                add_span_attributes(
                    tool_execution_success=True,
                    tool_output_length=len(str(result.output)) if result.output else 0,
                    has_debug_info=result.debug is not None
                )
                
                return result
                
            except Exception as e:
                record_exception(e)
                add_span_attributes(tool_execution_success=False)
                raise
    
    def register_custom_tool(self, tool: BaseTool) -> None:
        """Register a custom tool instance."""
        self.registry.register_tool(tool)
    
    def register_custom_function(
        self, 
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "custom"
    ) -> None:
        """Register a custom function as a tool."""
        self.registry.register_function(
            func, 
            name=name, 
            description=description, 
            category=category
        )
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        return self.registry.remove_tool(tool_name)
    
    def create_tool_functions_for_agent(
        self,
        tool_names: List[str],
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Callable]:
        """Create callable functions for PydanticAI agent integration."""
        import inspect
        from typing import get_type_hints
        
        functions = {}
        
        for tool_name in tool_names:
            # Get tool metadata to verify it exists
            metadata = self.get_tool_metadata(tool_name)
            if not metadata:
                continue
            
            # Create a wrapper function for the tool with proper signature
            def create_tool_wrapper(tool_name_inner, metadata_inner):
                # Build function signature dynamically
                params = []
                annotations = {}
                
                for param_name, param_info in metadata_inner.parameters.items():
                    if isinstance(param_info, dict):
                        # Extract type information
                        param_type = param_info.get("type", "string")
                        required = param_info.get("required", False)
                        default_val = param_info.get("default")
                        
                        # Convert string type to Python type
                        python_type = str  # Default
                        if param_type == "integer":
                            python_type = int
                        elif param_type == "number":
                            python_type = float
                        elif param_type == "boolean":
                            python_type = bool
                        elif param_type == "array":
                            python_type = list
                        elif param_type == "object":
                            python_type = dict
                        
                        annotations[param_name] = python_type
                        
                        # Create parameter with proper default
                        if required and default_val is None:
                            param = inspect.Parameter(
                                param_name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                annotation=python_type
                            )
                        else:
                            default_value = default_val if default_val is not None else None
                            param = inspect.Parameter(
                                param_name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=default_value,
                                annotation=python_type
                            )
                        params.append(param)
                
                # Create the async wrapper function with proper signature
                async def tool_wrapper(*args, **kwargs):
                    # Convert positional args to kwargs based on parameter names
                    param_names = list(metadata_inner.parameters.keys())
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            kwargs[param_names[i]] = arg
                    
                    # Merge base config with any tool-specific config
                    config = base_config.copy() if base_config else {}
                    if 'config' in kwargs:
                        config.update(kwargs.pop('config'))
                    
                    # Add operation context for tracing
                    kwargs['operation'] = 'agent_call'
                    
                    result = await self.execute_tool(tool_name_inner, config=config, **kwargs)
                    
                    # Return the output for successful executions, raise for errors
                    if result.status == "success":
                        return result.output
                    else:
                        raise Exception(f"Tool '{tool_name_inner}' failed: {result.message}")
                
                # Set proper function metadata
                tool_wrapper.__name__ = tool_name_inner
                tool_wrapper.__doc__ = metadata_inner.description
                tool_wrapper.__annotations__ = annotations
                
                # Create proper signature
                if params:
                    sig = inspect.Signature(params)
                    tool_wrapper.__signature__ = sig
                
                return tool_wrapper
            
            tool_wrapper = create_tool_wrapper(tool_name, metadata)
            functions[tool_name] = tool_wrapper
        
        return functions
    
    def get_tool_schemas_for_pydantic_ai(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Get tool schemas in PydanticAI format."""
        schemas = []
        
        for tool_name in tool_names:
            metadata = self.get_tool_metadata(tool_name)
            if not metadata:
                continue
            
            # Convert our tool metadata to PydanticAI tool schema format
            schema = {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert parameter definitions
            for param_name, param_info in metadata.parameters.items():
                if isinstance(param_info, dict):
                    schema["parameters"]["properties"][param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", "")
                    }
                    if param_info.get("required", False):
                        schema["parameters"]["required"].append(param_name)
                    if "default" in param_info:
                        schema["parameters"]["properties"][param_name]["default"] = param_info["default"]
            
            schemas.append(schema)
        
        return schemas


# Global tool service instance
_tool_service: Optional[ToolService] = None


def get_tool_service() -> ToolService:
    """Get the global tool service instance."""
    global _tool_service
    if _tool_service is None:
        _tool_service = ToolService()
    return _tool_service