from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, Part
from google.adk.models import LlmRequest, LlmResponse
from .handler_interface import ModelBeforeHandler
import logging

logger = logging.getLogger(__name__)

class EmptyPromptHandler(ModelBeforeHandler):
    """A handler that checks for and responds to empty prompts or 'TEST_EMPTY_MODEL'."""

    def handle(self, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmRequest | LlmResponse | None:
        """Checks if the prompt is empty (or specific test phrase) and returns a response if it is."""
        prompt_text = ""
        try:
            if llm_request.contents:
                last_content = llm_request.contents[-1]
                if last_content.parts:
                    prompt_text = " ".join(part.text for part in last_content.parts if part.text)

            # --- Combined Condition --- 
            is_empty_or_test = (not prompt_text or not prompt_text.strip()) or prompt_text.strip() == "TEST_EMPTY_MODEL"
            
            if is_empty_or_test:
                trigger = "empty prompt" if (not prompt_text or not prompt_text.strip()) else "'TEST_EMPTY_MODEL'"
                logger.warning(f"Intercepted {trigger} via before_model_callback. Preventing LLM call.")
                # Return an LlmResponse
                return LlmResponse(
                    content=Content(parts=[Part(text="I received an empty message. Please provide some input.")])
                )
            # --- END Combined Condition ---

        except Exception as e:
            logger.error(f"Error in EmptyPromptHandler (before_model_callback): {e}", exc_info=True)

        # Return None to indicate that this handler did not stop the processing
        return None
