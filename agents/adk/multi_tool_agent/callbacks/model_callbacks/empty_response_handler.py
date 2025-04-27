from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai.types import Content, Part
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmptyResponseHandler:
    """Handles empty LLM responses after the model call."""
    def handle(self, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        """Checks if the LLM response content is empty and returns a default message if so."""
        # Keep print at the very beginning for debugging
        print("DEBUG: Entering EmptyResponseHandler.handle")
        try:
            is_empty = True
            if llm_response and llm_response.content and llm_response.content.parts:
                response_text = "".join(part.text for part in llm_response.content.parts if part.text)
                if response_text and response_text.strip():
                    is_empty = False

            if is_empty:
                # Keep the print inside the condition as well
                print("DEBUG: LLM returned empty content. Overriding response via after_model_callback.")
                # Also try logging (uncommented)
                logger.warning("LLM returned empty content. Overriding response via after_model_callback.")
                # Return valid LlmResponse (no finish_reason)
                return LlmResponse(
                    content=Content(parts=[Part(text="The model returned an empty response. Can you try rephrasing?")])
                )
        except Exception as e:
            logger.error(f"Error in EmptyResponseHandler (after_model_callback): {e}", exc_info=True)

        return None # Let original response proceed
