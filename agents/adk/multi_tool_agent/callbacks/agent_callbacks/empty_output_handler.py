from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content, Part
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class EmptyOutputHandler:
    """Handles empty final agent output, checking if any output occurred during the turn."""

    def _is_content_empty(self, content: Optional[Content]) -> bool:
        """Helper function to check if Content is effectively empty."""
        if not content or not content.parts:
            return True
        output_text = "".join(part.text for part in content.parts if part.text)
        return not output_text or not output_text.strip()

    def handle(self, callback_context: CallbackContext, agent_output: Content) -> Optional[Content]:
        """Checks final agent output. If empty, checks history (via get_session) for prior output."""
        try:
            final_output_is_empty = self._is_content_empty(agent_output)

            if final_output_is_empty:
                # Final output is empty, check history using get_session
                invocation_context = callback_context._invocation_context
                session_service = invocation_context.session_service
                current_session_info = invocation_context.session

                retrieved_session = session_service.get_session(
                    app_name=current_session_info.app_name,
                    user_id=current_session_info.user_id,
                    session_id=current_session_info.id
                )

                if not retrieved_session or not hasattr(retrieved_session, 'events'):
                    logger.error("Could not retrieve session or session has no events attribute in EmptyOutputHandler.")
                    # Decide what to do - maybe still return the empty message?
                    # For now, let's assume something went wrong and don't override.
                    return None

                history = retrieved_session.events
                current_invocation_id = invocation_context.invocation_id
                agent_name = invocation_context.agent.name
                prior_output_found = False

                # Iterate backwards through history relevant to this invocation
                for event in reversed(history):
                    if event.invocation_id != current_invocation_id:
                        break
                    if event.author == agent_name:
                        if not self._is_content_empty(event.content):
                            prior_output_found = True
                            break

                if not prior_output_found:
                    logger.warning("Agent produced no meaningful output during the entire turn. Overriding final output.")
                    return Content(parts=[Part(text="The agent finished processing but produced no meaningful output.")])
                else:
                    logger.info("Agent finished turn with empty final Content, but prior output was sent.")
                    return None

        except AttributeError as ae:
             logger.error(f"AttributeError accessing history via get_session in EmptyOutputHandler: {ae}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Error in EmptyOutputHandler (after_agent_callback): {e}", exc_info=True)
            return None

        return None
