import json
from typing import List, Optional, Dict, Any, Generator
from src.core.api_client import api_client
from src.services.adk_backend.adk_models import AgentRunRequestADK, SessionADK, EventADK, ContentInputADK, MessagePartADK
from src.shared.models.common_models import MessagePart # Import common MessagePart
# APP_NAME is now passed dynamically as agent_name

class ADKBackendClient:
    """
    Low-level client for interacting with the ADK backend API.
    Handles direct mapping to ADK endpoints and uses ADK-specific models.
    """

    def list_apps(self) -> List[str]:
        response = api_client.get("/list-apps")
        return response.json()

    def create_session(self, user_id: str, agent_name: str, session_id: Optional[str] = None, initial_state: Optional[Dict[str, Any]] = None) -> SessionADK:
        endpoint = f"/apps/{agent_name}/users/{user_id}/sessions"
        if session_id:
            endpoint += f"/{session_id}"
        
        payload = {"state": initial_state} if initial_state else {}
        response = api_client.post(endpoint, json=payload)
        return SessionADK(**response.json())

    def _prepare_payload(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart], streaming: bool) -> AgentRunRequestADK:
        adk_message_parts = []
        for part in message_parts:
            adk_part_kwargs = {
                "text": part.text,
                "inlineData": None,
                "functionCall": part.function_call,
                "functionResponse": part.function_response
            }

            if part.image_url:
                if "base64," in part.image_url:
                    mime_type = part.image_url.split(';')[0].split(':')[1]
                    data = part.image_url.split(',', 1)[1]
                    adk_part_kwargs["inlineData"] = {"mimeType": mime_type, "data": data}
            elif part.file_data:
                adk_part_kwargs["inlineData"] = {
                    "mimeType": part.file_data.get("mime_type", "application/octet-stream"),
                    "data": part.file_data.get("data", "")
                }
            
            adk_message_parts.append(MessagePartADK(**adk_part_kwargs))

        return AgentRunRequestADK(
            appName=agent_name,
            userId=user_id,
            sessionId=session_id,
            newMessage=ContentInputADK(role="user",parts=adk_message_parts),
            streaming=streaming
        )

    def send_message_list(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> List[EventADK]:
        payload = self._prepare_payload(user_id, agent_name, session_id, message_parts, streaming=False)
        response = api_client.post("/run", json=payload.model_dump(by_alias=True))
        return [EventADK(**event_data) for event_data in response.json()]

    def send_message_stream(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> Generator[EventADK, None, None]:
        payload = self._prepare_payload(user_id, agent_name, session_id, message_parts, streaming=True)
        response = api_client.post("/run_sse", json=payload.model_dump(by_alias=True), stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    try:
                        event_data = json.loads(decoded_line[len("data:"):])
                        yield EventADK(**event_data)
                    except json.JSONDecodeError:
                        pass

    def get_session_history(self, user_id: str, agent_name: str, session_id: str) -> SessionADK:
        endpoint = f"/apps/{agent_name}/users/{user_id}/sessions/{session_id}"
        response = api_client.get(endpoint)
        return SessionADK(**response.json())

    def list_sessions(self, user_id: str, agent_name: str) -> List[SessionADK]:
        """Lists all sessions for a given user and agent."""
        endpoint = f"/apps/{agent_name}/users/{user_id}/sessions"
        response = api_client.get(endpoint)
        return [SessionADK(**session_data) for session_data in response.json()]

    def delete_session(self, user_id: str, agent_name: str, session_id: str):
        endpoint = f"/apps/{agent_name}/users/{user_id}/sessions/{session_id}"
        api_client.delete(endpoint)
        return True # Indicate success

# Instantiate the ADK backend client
adk_backend_client = ADKBackendClient()