from typing import Protocol, List, Optional, Dict, Any, Generator
from src.shared.models.common_models import Conversation, Message, MessagePart

class BackendClient(Protocol):
    """
    Defines the common interface that all backend clients must adhere to.
    This allows the UI to interact with any backend in a consistent way.
    """
    def list_apps(self) -> List[str]:
        ...

    def create_session(self, user_id: str, agent_name: str, session_id: Optional[str] = None, initial_state: Optional[Dict[str, Any]] = None) -> Conversation:
        ...

    def send_message_list(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> List[Message]:
        ...

    def send_message_stream(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> Generator[Message, None, None]:
        ...

    def get_session_history(self, user_id: str, agent_name: str, session_id: str) -> Conversation:
        ...

    def list_sessions(self, user_id: str, agent_name: str) -> List[Conversation]:
        ...

    def delete_session(self, user_id: str, agent_name: str, session_id: str) -> bool:
        ...