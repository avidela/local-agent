from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content
from typing import Optional
import logging

# Import specific handler classes
from .empty_input_handler import EmptyInputHandler
from .empty_output_handler import EmptyOutputHandler

logger = logging.getLogger(__name__)

# Instantiate handlers
before_handlers = [EmptyInputHandler()]
after_handlers = [EmptyOutputHandler()]

def handle_before_agent_callback(callback_context: CallbackContext) -> Optional[Content]:
    """Iterates through registered before_agent handlers."""
    for handler in before_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            try:
                result = handler.handle(callback_context)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Error in before_agent_callback handler {type(handler).__name__}: {e}", exc_info=True)
    return None

# Define handle_after_agent_callback to accept ONLY context, matching the framework call signature
def handle_after_agent_callback(callback_context: CallbackContext) -> Optional[Content]:
    """Iterates through registered after_agent handlers.
    Retrieves agent output from context history via get_session as ADK only passes context.
    """
    agent_output: Optional[Content] = None
    original_agent_output: Optional[Content] = None # Keep track of the original
    try:
        # Attempt to retrieve the agent's output via get_session
        invocation_context = callback_context._invocation_context
        session_service = invocation_context.session_service
        current_session_info = invocation_context.session

        retrieved_session = session_service.get_session(
            app_name=current_session_info.app_name,
            user_id=current_session_info.user_id,
            session_id=current_session_info.id
        )

        if retrieved_session and hasattr(retrieved_session, 'events') and retrieved_session.events:
            history = retrieved_session.events
            agent_name = invocation_context.agent.name
            # Check the last event
            if history[-1].author == agent_name:
                 agent_output = history[-1].content
                 original_agent_output = agent_output # Store the original
            else:
                 logger.info(
                     "Last event in history not from current agent in handle_after_agent_callback. "
                     "Last author: %s, Agent: %s",
                     history[-1].author, agent_name
                 )
        else:
             logger.warning(
                 "Could not retrieve session with events in handle_after_agent_callback."
             )
             # Proceed with agent_output as None

    except AttributeError as ae:
        logger.error(f"AttributeError retrieving context/session info in handle_after_agent_callback: {ae}", exc_info=True)
    except Exception as e:
        logger.error(f"Error retrieving agent_output via get_session in handle_after_agent_callback: {e}", exc_info=True)

    # Now iterate through handlers with the retrieved agent_output (which might be None)
    current_output = agent_output
    for handler in after_handlers:
        if hasattr(handler, 'handle') and callable(handler.handle):
            try:
                # Pass context and the retrieved output to the handler
                result = handler.handle(callback_context, current_output)
                if result is not None:
                    current_output = result
            except Exception as e:
                logger.error(f"Error in after_agent_callback handler {type(handler).__name__}: {e}", exc_info=True)

    # Return the final output (original or modified)
    if current_output is original_agent_output:
        return None # No changes made, or retrieval failed initially
    else:
        return current_output # Return modified output
