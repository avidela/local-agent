import streamlit as st
import uuid # For generating unique session IDs
from typing import Optional, List, Any
from src.modules.conversations.services.conversation_service import conversation_service
from src.modules.conversations.components.message_display import message_display
from src.modules.conversations.components.chat_input import chat_input_component
from src.shared.models.common_models import Conversation, Message, MessagePart
import base64 # For encoding file data

def conversations_page():
    st.title("Conversations")

    # Initialize session state variables if they don't exist
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user" # Or get from a login system
    if "message_processed" not in st.session_state:
        st.session_state.message_processed = False

    # --- Conversation Selection/Creation (Simplified for now, will be in sidebar) ---
    # For now, let's assume a single conversation or a way to select it.
    # In a full app, this would be driven by the sidebar_controls.py
    
    # If no conversation is active, create a new one
    # If no conversation is active, create a new one
    if st.session_state.current_conversation_id is None:
        if st.button("Start New Conversation", key="start_new_conversation_button"):
            # Ensure an app is selected before creating a new conversation
            if st.session_state.selected_adk_app:
                new_session = conversation_service.create_new_session(
                    user_id=st.session_state.user_id,
                    agent_name=st.session_state.selected_adk_app,
                    session_id=str(uuid.uuid4())
                )
                st.session_state.current_conversation_id = new_session.id
                st.session_state.messages = [] # Clear messages for new conversation
                st.success(f"New conversation started with {st.session_state.selected_adk_app}: {new_session.id}")
                # No st.rerun() here; state change will trigger it.
            else:
                st.warning("Please select an agent from the sidebar to start a new conversation.")

    # Define the message sending handler outside the conditional block
    def handle_send_message(user_message_text: str, uploaded_files: Optional[List[Any]] = None):
        if not user_message_text and not uploaded_files:
            return # Don't send empty messages

        # Ensure a conversation is active and an agent is selected
        if not st.session_state.current_conversation_id or not st.session_state.selected_adk_app:
            st.warning("Please start a new conversation and select an agent from the sidebar.")
            return

        message_parts = []
        if user_message_text:
            message_parts.append(MessagePart(text=user_message_text))
        
        if uploaded_files: # Check if uploaded_files is not None before iterating
            for file in uploaded_files:
                file_bytes = file.read()
                encoded_file = base64.b64encode(file_bytes).decode('utf-8')
                
                message_parts.append(MessagePart(file_data={
                    "mime_type": file.type,
                    "data": encoded_file,
                    "display_name": file.name
                }))
        
        # Add user message (and files) to display immediately
        st.session_state.messages.append(Message(author="user", parts=message_parts))
        
        # Send message to agent
        try:
            is_streaming = st.session_state.get("streaming_enabled", False)
            
            if is_streaming:
                # Define a generator function to yield text chunks for st.write_stream
                def stream_response_chunks():
                    for message_chunk in conversation_service.send_message_to_agent(
                        user_id=st.session_state.user_id,
                        agent_name=st.session_state.selected_adk_app,
                        session_id=st.session_state.current_conversation_id,
                        message_parts=message_parts,
                        streaming=True
                    ):
                        # Assuming message_chunk is a Message object with a single text part
                        if message_chunk.parts and message_chunk.parts[0].text:
                            yield message_chunk.parts[0].text
                
                # Use st.write_stream to display the streamed content
                # st.write_stream returns the full accumulated content
                full_response_content = st.write_stream(stream_response_chunks())
                
                # After streaming is complete, add the full response to session state
                st.session_state.messages.append(Message(author="agent", parts=[MessagePart(text=str(full_response_content))]))
            else:
                response_messages = conversation_service.send_message_to_agent(
                    user_id=st.session_state.user_id,
                    agent_name=st.session_state.selected_adk_app,
                    session_id=st.session_state.current_conversation_id,
                    message_parts=message_parts,
                    streaming=False
                )
                st.session_state.messages.extend(response_messages)
        except Exception as e:
            st.error(f"Failed to get response from agent: {e}")
        
        st.session_state.message_processed = True # Set flag to trigger rerun outside callback

    if st.session_state.current_conversation_id:
        st.subheader(f"Conversation ID: {st.session_state.current_conversation_id}")

        # Display messages
        message_display(
            st.session_state.messages,
            show_author=st.session_state.show_author,
            auto_collapse_tool_details=st.session_state.auto_collapse_tool_details
        )

        # Render the chat input component
        chat_input_component(on_send=handle_send_message, disabled=st.session_state.selected_adk_app is None)

    else:
        st.info("Start a new conversation to begin interacting with the agent.")
    
    # Check flag to trigger rerun after callback completes
    if st.session_state.message_processed:
        st.session_state.message_processed = False # Reset the flag
        st.rerun()

# This is how Streamlit's multi-page app works.
# The main app.py will call this function when this page is selected.
if __name__ == "__main__":
    conversations_page()