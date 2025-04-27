from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

from .model_callbacks.empty_prompt_handler import EmptyPromptHandler

def create_model_before_handler(handlers: list):
    """Factory function to create the combined before_model callback handler."""

    def handle_before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmRequest | LlmResponse | None:
        """Generic callback to handle actions before sending prompt to the LLM."""
        # Iterate through the registered handlers and apply them
        for handler in handlers:
            result = handler.handle(callback_context, llm_request)
            # If a handler returns a non-None result (indicating it handled the request),
            # return that result immediately to stop further processing.
            if result is not None:
                return result

        # If no handler intercepted, return the original request to continue the flow to the LLM
        return llm_request

    return handle_before_model

# Define the list of handlers to be applied
# In a more complex scenario, this list could be dynamically built based on configuration
model_before_handlers = [
    EmptyPromptHandler(),
    # Add other model before handlers here in the future
]

# Create the actual callback function using the factory
# This is the function that will be registered with the agent
handle_before_model_callback = create_model_before_handler(model_before_handlers)
