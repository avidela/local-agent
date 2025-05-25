from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class EmptyResultHandler:
    """Handles empty tool results after a tool call."""
    def handle(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        tool_context: ToolContext,
        tool_response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Checks if the tool response dictionary is empty or lacks meaningful content."""
        try:
            # *** Crucial: Ignore the internal transfer_to_agent tool ***
            if tool.name == "transfer_to_agent":
                return None # Do not modify the result of agent transfers

            is_empty = True
            # --- Adapt this check based on your tools' return structure ---
            if tool_response:
                # Example: Check if *any* value in the dict is non-empty
                for value in tool_response.values():
                    if isinstance(value, str) and value.strip():
                        is_empty = False
                        break
                    elif isinstance(value, (int, float)) and value != 0:
                        is_empty = False
                        break
                    elif isinstance(value, (list, dict)) and value:
                        is_empty = False
                        break
                    # Add other checks as needed
            # --- End of adaptable check ---

            if is_empty:
                logger.warning(f"Tool '{tool.name}' returned an empty result. Modifying response via after_tool_callback.")
                return {"status": f"Tool '{tool.name}' executed but returned no meaningful content."}

        except Exception as e:
            logger.error(f"Error in EmptyResultHandler (after_tool_callback): {e}", exc_info=True)

        return None # Let original tool_response proceed
