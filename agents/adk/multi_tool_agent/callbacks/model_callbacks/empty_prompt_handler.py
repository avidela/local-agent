from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, Part
from google.adk.models import LlmRequest, LlmResponse
from .handler_interface import ModelBeforeHandler

class EmptyPromptHandler(ModelBeforeHandler):
    """A handler that checks for and responds to empty prompts."""

    def handle(self, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmRequest | LlmResponse | None:
        """Checks if the prompt is empty and returns a response if it is."""
        prompt_text = ""
        if llm_request.contents:
            # Assuming the prompt is primarily in the first part of the last content item
            # Adjust this logic based on how prompts are actually structured in your requests
            last_content = llm_request.contents[-1]
            if last_content.parts:
                # Concatenate text from all parts in the last content block
                prompt_text = " ".join(part.text for part in last_content.parts if part.text)

        if not prompt_text or not prompt_text.strip():
            print("Intercepted empty or whitespace-only prompt. Preventing LLM call.")
            # Return an LlmResponse to stop the flow and provide a specific output
            return LlmResponse(
                content=Content(parts=[Part(text="I received an empty message. Please provide some input.")]),
                # Optionally, set a finish reason to indicate why the LLM was not called
                finish_reason="STOP"
            )

        # Return None to indicate that this handler did not stop the processing
        return None
