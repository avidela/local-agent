# Agent Callbacks

Callbacks provide a mechanism to hook into various stages of the agent's request processing lifecycle within the Agent Development Kit (ADK). They allow developers to inspect, modify, or even intercept operations before or after they occur.

## Purpose

Callbacks enable advanced customization and control over the agent's behavior. Common use cases include:

*   **Logging:** Recording detailed information at specific lifecycle points.
*   **Input/Output Validation:** Checking or sanitizing data before it's processed or returned.
*   **Error Handling:** Implementing custom logic for specific error conditions (e.g., empty responses, empty tool results).
*   **Flow Control:** Modifying requests/responses or even skipping certain steps (like an LLM call or tool execution) based on context.
*   **Adding Custom Logic:** Injecting specific actions at defined stages (e.g., formatting results, adding metadata).

## Callback Types

The ADK agent defines several callback points, registered via arguments in the `Agent` constructor:

*   **`before_agent_callback`:** Executed before the agent starts processing a user request or turn.
    *   Receives: `CallbackContext`
    *   Can return: `Content` (to skip the agent's main logic and return this content directly) or `None`.
*   **`after_agent_callback`:** Executed after the agent has finished processing and generated its final output for the turn.
    *   Receives: `CallbackContext` (Note: Agent output is *not* directly passed; handlers must retrieve it from history via context if needed).
    *   Can return: `Content` (to replace the agent's original final output) or `None`.
*   **`before_model_callback`:** Executed just before the underlying language model (LLM) is called.
    *   Receives: `CallbackContext`, `LlmRequest`
    *   Can return: `LlmResponse` (to skip the LLM call and use this response instead) or `None`.
*   **`after_model_callback`:** Executed immediately after the LLM returns a response.
    *   Receives: `CallbackContext`, `LlmResponse`
    *   Can return: `LlmResponse` (to replace the original LLM response) or `None`.
*   **`before_tool_callback`:** Executed before a specific tool function is called.
    *   Receives: `BaseTool`, `Dict[str, Any]` (args), `ToolContext`
    *   Can return: `Dict[str, Any]` (to skip the tool call and use this dict as the result) or `None`.
*   **`after_tool_callback`:** Executed after a tool function returns its result.
    *   Receives: `BaseTool`, `Dict[str, Any]` (args), `ToolContext`, `Dict[str, Any]` (tool_response)
    *   Can return: `Dict[str, Any]` (to replace the original tool response) or `None`.

## `CallbackContext`

All callback functions receive an instance of `google.adk.agents.callback_context.CallbackContext`. This object provides access to the current state and context of the agent's invocation. Handlers often need to access internal attributes like `callback_context._invocation_context` to retrieve session services, session IDs, invocation IDs, agent names, etc., especially for accessing historical events when the callback signature doesn't directly provide the needed data (e.g., `after_agent_callback`).

## Implementation Pattern

This project follows a pattern for organizing and registering callbacks:

1.  **Define Handler Classes:** Create specific classes (e.g., `EmptyInputHandler`) that contain the logic for a particular check or modification. These classes typically have a `handle` method matching the expected signature for their target callback type. Abstract base classes (like `ModelBeforeHandler`) can be used for interface consistency.
2.  **Organize Handlers:** Place handler classes in the `local-agent/agents/adk/multi_tool_agent/callbacks/` directory, usually within subdirectories corresponding to the callback type (`agent_callbacks`, `model_callbacks`, `tool_callbacks`).
3.  **Orchestrate Handlers:** In the `__init__.py` file for each callback type subdirectory (e.g., `callbacks/agent_callbacks/__init__.py`), import the handler classes, instantiate them, and create lists (e.g., `before_handlers`, `after_handlers`). Define central `handle_...` functions (e.g., `handle_before_agent_callback`) that iterate through the corresponding list of handlers, calling their `handle` methods and managing the return values (e.g., returning early if a handler intercepts, or passing modified results to the next handler).
4.  **Register Central Handlers:** In the main agent definition file (`local-agent/agents/adk/multi_tool_agent/agent.py`), import the central `handle_...` functions from the top-level `callbacks/__init__.py` and pass them to the corresponding arguments (`before_agent_callback`, `after_model_callback`, etc.) in the `Agent` constructor.

```python
# Example registration in agent.py
from .callbacks import (
    handle_before_agent_callback,
    handle_after_agent_callback,
    # ... other handlers
)
# ...
root_agent = Agent(
    # ... other config
    before_agent_callback=handle_before_agent_callback,
    after_agent_callback=handle_after_agent_callback,
    # ... other callbacks
)
```

## Examples in this Project

This project utilizes callbacks primarily for handling empty or potentially problematic inputs/outputs, making the agent more robust:

*   `EmptyInputHandler` (`before_agent_callback`): Intercepts empty user input or specific test strings, returning a default message.
*   `EmptyPromptHandler` (`before_model_callback`): Intercepts empty prompts to the LLM or specific test strings, returning a default `LlmResponse` to skip the LLM call. *(Note: Currently disabled by default in `agent.py`)*.
*   `EmptyResponseHandler` (`after_model_callback`): Replaces empty responses from the LLM with a default message.
*   `EmptyOutputHandler` (`after_agent_callback`): Checks if the agent's final output is empty *and* if any output was produced earlier in the turn (using session history). If no output occurred at all, it provides a default final message.
*   `EmptyResultHandler` (`after_tool_callback`): Replaces empty results from tools (except `transfer_to_agent`) with a status message dictionary.

