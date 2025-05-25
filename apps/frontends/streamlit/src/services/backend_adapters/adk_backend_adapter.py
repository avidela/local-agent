from typing import List, Optional, Dict, Any, Union, Generator
from src.shared.models.common_models import Conversation, Message, MessagePart
from src.services.adk_backend.adk_api_client import adk_backend_client
from src.services.adk_backend.adk_models import SessionADK, EventADK, MessagePartADK
from src.services.backend_client_protocol import BackendClient # Import BackendClient protocol

class ADKBackendAdapter(BackendClient):
    """
    Adapts the ADKBackendClient to the generic BackendClient protocol.
    Handles mapping between ADK-specific models and common models.
    """
    def list_apps(self) -> List[str]:
        return adk_backend_client.list_apps()

    def _convert_adk_session_to_common(self, adk_session: SessionADK) -> Conversation:
        messages = []
        for event in adk_session.events:
            author = event.author
            parts = []
            if event.content and event.content.parts:
                for part in event.content.parts:
                    text_content = str(part.text) if part.text is not None else None
                    # Assuming ADK's inlineData maps to our file_data/image_url
                    image_url = None
                    video_url = None # Initialize video_url
                    file_data = None
                    if hasattr(part, 'inlineData') and part.inlineData:
                        mime_type = part.inlineData.get("mimeType")
                        data = part.inlineData.get("data")
                        display_name = part.inlineData.get("displayName")
                        if mime_type and data:
                            if "image" in mime_type:
                                image_url = f"data:{mime_type};base64,{data}"
                            elif "video" in mime_type: # Handle video MIME types
                                video_url = f"data:{mime_type};base64,{data}"
                            else:
                                file_data = {"mime_type": mime_type, "data": data, "display_name": display_name}
                    
                    parts.append(MessagePart(text=text_content, image_url=image_url, video_url=video_url, file_data=file_data))
            
            is_tool_call = False
            tool_name = None
            tool_args = None
            is_tool_response = False
            tool_response_content = None

            if event.actions and "toolUses" in event.actions and event.actions["toolUses"]:
                # Assuming a single tool use per event for simplicity in this adapter
                tool_use = event.actions["toolUses"][0]
                is_tool_call = True
                tool_name = tool_use.get("name")
                tool_args = tool_use.get("args")
                author = "tool_call" # Or a specific author for tool calls

            # ADK's FunctionResponse is part of content, not actions directly
            # This mapping might need refinement based on how ADK structures tool responses
            # For now, let's assume tool responses are just text content from the agent
            # or explicitly marked in the event.
            # If ADK provides a distinct 'functionResponse' part type, use that.

            messages.append(Message(
                author=author,
                parts=parts,
                timestamp=event.timestamp,
                is_tool_call=is_tool_call,
                tool_name=tool_name,
                tool_args=tool_args,
                is_tool_response=is_tool_response,
                tool_response_content=tool_response_content
            ))
        
        return Conversation(
            id=adk_session.id,
            user_id=adk_session.userId,
            app_name=adk_session.appName,
            messages=messages,
            state=adk_session.state,
            last_update_time=adk_session.lastUpdateTime
        )

    def create_session(self, user_id: str, agent_name: str, session_id: Optional[str] = None, initial_state: Optional[Dict[str, Any]] = None) -> Conversation:
        adk_session = adk_backend_client.create_session(user_id, agent_name, session_id, initial_state)
        return self._convert_adk_session_to_common(adk_session)

    def send_message_list(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> List[Message]:
        adk_response_list = adk_backend_client.send_message_list(user_id, agent_name, session_id, message_parts)
        return [self._convert_adk_event_to_common_message(event) for event in adk_response_list]

    def send_message_stream(self, user_id: str, agent_name: str, session_id: str, message_parts: List[MessagePart]) -> Generator[Message, None, None]:
        adk_response_generator = adk_backend_client.send_message_stream(user_id, agent_name, session_id, message_parts)
        
        def message_generator():
            for event in adk_response_generator:
                yield self._convert_adk_event_to_common_message(event)
        return message_generator()

    def _convert_adk_event_to_common_message(self, event: EventADK) -> Message:
        author = event.author
        parts = []
        if event.content and event.content.parts:
            for part in event.content.parts:
                text_content = str(part.text) if part.text is not None else None
                image_url = None
                video_url = None # Initialize video_url
                file_data = None
                if hasattr(part, 'inlineData') and part.inlineData:
                    mime_type = part.inlineData.get("mimeType")
                    data = part.inlineData.get("data")
                    display_name = part.inlineData.get("displayName")
                    if mime_type and data:
                        if "image" in mime_type:
                            image_url = f"data:{mime_type};base64,{data}"
                        elif "video" in mime_type: # Handle video MIME types
                            video_url = f"data:{mime_type};base64,{data}"
                        else:
                            file_data = {"mime_type": mime_type, "data": data, "display_name": display_name}
                
                parts.append(MessagePart(text=text_content, image_url=image_url, video_url=video_url, file_data=file_data))
        
        is_tool_call = False
        tool_name = None
        tool_args = None
        is_tool_response = False
        tool_response_content = None

        if event.actions and "toolUses" in event.actions and event.actions["toolUses"]:
            tool_use = event.actions["toolUses"][0]
            is_tool_call = True
            tool_name = tool_use.get("name")
            tool_args = tool_use.get("args")
            author = "tool_call" # Or a specific author for tool calls

        return Message(
            author=author,
            parts=parts,
            timestamp=event.timestamp,
            is_tool_call=is_tool_call,
            tool_name=tool_name,
            tool_args=tool_args,
            is_tool_response=is_tool_response,
            tool_response_content=tool_response_content
        )

    def get_session_history(self, user_id: str, agent_name: str, session_id: str) -> Conversation:
        adk_session = adk_backend_client.get_session_history(user_id, agent_name, session_id)
        return self._convert_adk_session_to_common(adk_session)

    def list_sessions(self, user_id: str, agent_name: str) -> List[Conversation]:
        """Lists all sessions for a given user and agent."""
        adk_sessions = adk_backend_client.list_sessions(user_id, agent_name)
        return [self._convert_adk_session_to_common(session) for session in adk_sessions]

    def delete_session(self, user_id: str, agent_name: str, session_id: str) -> bool:
        return adk_backend_client.delete_session(user_id, agent_name, session_id)