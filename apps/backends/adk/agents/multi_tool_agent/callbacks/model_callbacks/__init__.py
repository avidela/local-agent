from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from typing import Optional

# Import specific handler classes
from .empty_prompt_handler import EmptyPromptHandler # Handles before_model
from .empty_response_handler import EmptyResponseHandler # Handles after_model

# Instantiate handlers
before_handlers = [EmptyPromptHandler()]
after_handlers = [EmptyResponseHandler()]

def handle_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Iterates through registered before_model handlers."""
    for handler in before_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            result = handler.handle(callback_context, llm_request)
            # If any handler returns a response, return it immediately
            if result is not None:
                return result
    return None # Proceed with model call

def handle_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Iterates through registered after_model handlers."""
    current_response = llm_response
    for handler in after_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            result = handler.handle(callback_context, current_response)
            # If a handler returns a new response, use it for the next handler or as final output
            if result is not None:
                current_response = result

    # Return the final response (original or modified)
    if current_response is llm_response:
        return None # No changes made
    else:
        return current_response # Return modified response
