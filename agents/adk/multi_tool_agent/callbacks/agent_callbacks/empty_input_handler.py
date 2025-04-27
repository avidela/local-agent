from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, Part
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmptyInputHandler:
    """Handles empty user input or 'TEST_EMPTY' before the agent's main logic runs."""
    def handle(self, callback_context: CallbackContext) -> Optional[Content]:
        """Checks the last user message in history. Returns Content to skip agent run if empty or 'TEST_EMPTY'."""
        try:
            invocation_context = callback_context._invocation_context
            session_service = invocation_context.session_service
            current_session_info = invocation_context.session

            retrieved_session = session_service.get_session(
                app_name=current_session_info.app_name,
                user_id=current_session_info.user_id,
                session_id=current_session_info.id
            )

            if not retrieved_session or not hasattr(retrieved_session, 'events'):
                 logger.error("Could not retrieve session or session has no events attribute.")
                 return None

            history = retrieved_session.events
            last_user_event = None
            for event in reversed(history):
                if event.author == 'user':
                    last_user_event = event
                    break

            input_text = ""
            if last_user_event and last_user_event.content and last_user_event.content.parts:
                input_text = "".join(part.text for part in last_user_event.content.parts if part.text)

            # --- Combined Condition --- 
            is_empty_or_test = (not input_text or not input_text.strip()) or input_text.strip() == "TEST_EMPTY"
            
            if is_empty_or_test:
                trigger = "empty user input" if (not input_text or not input_text.strip()) else "'TEST_EMPTY'"
                logger.warning(f"Intercepted {trigger} via before_agent_callback. Skipping agent run.")
                return Content(parts=[Part(text="I received an empty message. Please provide some input.")])
            # --- END Combined Condition ---

        except AttributeError as ae:
             logger.error(f"AttributeError accessing history via get_session in EmptyInputHandler: {ae}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Error in EmptyInputHandler (before_agent_callback): {e}", exc_info=True)
            return None

        return None
