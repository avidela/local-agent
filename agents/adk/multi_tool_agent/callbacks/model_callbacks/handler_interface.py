import abc
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

class ModelBeforeHandler(abc.ABC):
    """Abstract base class for handlers that process requests before the LLM call."""

    @abc.abstractmethod
    def handle(self, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmRequest | LlmResponse | None:
        """Handle the request before it is sent to the LLM.

        Args:
            callback_context: The context for the callback.
            llm_request: The request object for the LLM.

        Returns:
            - The original or modified LlmRequest to continue processing.
            - An LlmResponse to stop processing and return a specific response.
            - None to indicate that this handler did not intercept and processing should continue to the next handler.
        """
        pass
