import streamlit as st
from typing import List, Optional, Dict, Any, Union, Generator
from src.shared.models.common_models import Conversation, Message, MessagePart
from src.services.backend_selector import active_backend_client

class ConversationService:
    """
    Provides a unified interface for managing conversations and interacting with the active backend.
    This service uses the abstract BackendClient interface, making it backend-agnostic.
    """
    
    def list_available_apps(self) -> List[str]:
        """Lists available agent applications from the active backend."""
        return active_backend_client.list_apps()

    def list_sessions(self, user_id: str, agent_name: str) -> List[Conversation]:
        """Lists all conversation sessions for a given user and agent."""
        # The backend client returns common Conversation models directly
        return active_backend_client.list_sessions(user_id, agent_name)

    def create_new_session(self, user_id: str, agent_name: str, session_id: Optional[str] = None, initial_state: Optional[Dict[str, Any]] = None) -> Conversation:
        """Creates a new conversation session with the active backend."""
        return active_backend_client.create_session(user_id, agent_name, session_id, initial_state)

    def send_message_to_agent(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart], streaming: bool = False) -> Union[List[Message], Generator[Message, None, None]]:
        """Sends a message (potentially multimodal) to the agent via the active backend and returns response messages."""
        if streaming:
            # When streaming, active_backend_client.send_message_stream returns a generator
            return active_backend_client.send_message_stream(user_id, agent_name, session_id, message_parts)
        else:
            # When not streaming, active_backend_client.send_message_list returns a list
            return active_backend_client.send_message_list(user_id, agent_name, session_id, message_parts)

    def get_conversation_history(self, user_id: str, agent_name: str, session_id: str) -> Conversation:
        """Retrieves the full conversation history for a given session from the active backend."""
        return active_backend_client.get_session_history(user_id, agent_name, session_id)

    def delete_conversation(self, user_id: str, agent_name: str, session_id: str) -> bool:
        """Deletes a conversation session from the active backend."""
        return active_backend_client.delete_session(user_id, agent_name, session_id)

# Instantiate the conversation service
conversation_service = ConversationService()