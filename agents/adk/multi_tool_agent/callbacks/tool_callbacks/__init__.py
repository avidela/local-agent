from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Any, Optional, Dict

# Import specific handler classes
from .empty_result_handler import EmptyResultHandler # Handles after_tool
# Add imports for before_tool handlers here if needed in the future

# Instantiate handlers
before_handlers = [] # No before_tool handlers defined yet
after_handlers = [EmptyResultHandler()]

def handle_before_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
    """Iterates through registered before_tool handlers."""
    for handler in before_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            result = handler.handle(tool, args, tool_context)
            # If any handler returns a dict, return it immediately to skip the tool call
            if result is not None:
                return result
    return None # Proceed with tool call

def handle_after_tool_callback(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Iterates through registered after_tool handlers."""
    current_response = tool_response
    for handler in after_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            result = handler.handle(tool, args, tool_context, current_response)
            # If a handler returns a new dict, use it for the next handler or as final output
            if result is not None:
                current_response = result

    # Return the final response (original or modified)
    if current_response is tool_response:
        return None # No changes made
    else:
        return current_response # Return modified response
