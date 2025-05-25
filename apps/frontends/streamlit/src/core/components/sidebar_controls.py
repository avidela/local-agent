import streamlit as st
from src.services.backend_selector import get_active_backend_client # Import the function, not the instance
from src.modules.conversations.services.conversation_service import conversation_service # For listing apps

def sidebar_controls():
    st.sidebar.header("Controls")

    # --- Backend Framework Selection ---
    available_frameworks = ["ADK"] # Extend this list as you add more backends (LangGraph, Pydantic-AI, Crew-AI)
    st.sidebar.selectbox(
        "Select Backend Framework",
        options=available_frameworks,
        key="selected_backend" # This key is read by backend_selector.py
    )
    # The value is automatically stored in st.session_state.selected_backend by the widget.

    # --- Agent Selection (Specific to the selected backend framework) ---
    # This section will dynamically change based on `selected_framework`
    if st.session_state.selected_backend == "ADK":
        available_adk_apps = []
        try:
            current_backend_client = get_active_backend_client()
            available_adk_apps = current_backend_client.list_apps()
            if not available_adk_apps:
                st.sidebar.warning("No ADK agents found. Is the backend running and configured?")
                available_adk_apps = ["No agents available"] # Provide a placeholder
        except Exception as e:
            st.sidebar.error(f"Error fetching ADK agents: {e}")
            available_adk_apps = ["Error loading agents"] # Indicate an error state

        st.sidebar.selectbox(
            "Select Agent (ADK App)",
            options=available_adk_apps,
            key="selected_adk_app"
        )
        # The value is automatically stored in st.session_state.selected_adk_app by the widget.
    else:
        # For other backends, you might have different selection mechanisms or just a placeholder
        st.session_state.selected_adk_app = None # Clear ADK app if another backend is selected
        st.sidebar.info(f"Agent selection for {st.session_state.selected_backend} backend not yet implemented.")

    # --- Conversation Management ---
    st.sidebar.subheader("Conversation Management")


    # List and select existing conversations
    if st.session_state.get("selected_adk_app"):
        user_id = st.session_state.user_id
        agent_name = st.session_state.selected_adk_app
        
        try:
            # Fetch existing sessions for the current user and agent
            # Ensure agent_name is not None before calling list_sessions
            if agent_name:
                existing_sessions = conversation_service.list_sessions(user_id, agent_name)
            else:
                existing_sessions = [] # No sessions if no agent selected
            
            session_options = ["-- Select or Start New --"] + [session.id for session in existing_sessions]
            
            # Find the index of the current conversation if it exists in the list
            current_session_index = 0
            if st.session_state.current_conversation_id and st.session_state.current_conversation_id in [s.id for s in existing_sessions]:
                # Find the index of the current conversation in the list of session IDs
                current_session_index = [s.id for s in existing_sessions].index(st.session_state.current_conversation_id) + 1 # +1 because of "-- Select or Start New --"

            selected_session_id = st.sidebar.selectbox(
                "Resume Conversation",
                options=session_options,
                index=current_session_index,
                key="selected_conversation_to_resume"
            )

            if selected_session_id != "-- Select or Start New --" and selected_session_id != st.session_state.current_conversation_id:
                # A new conversation was selected from the dropdown
                st.session_state.current_conversation_id = selected_session_id
                
                # Ensure agent_name is not None before calling get_conversation_history
                if st.session_state.selected_adk_app:
                    # Fetch and load messages for the selected conversation
                    conversation_history = conversation_service.get_conversation_history(
                        user_id=st.session_state.user_id,
                        agent_name=st.session_state.selected_adk_app,
                        session_id=selected_session_id
                    )
                    st.session_state.messages = conversation_history.messages
                else:
                    st.warning("Please select an agent before resuming a conversation.")
                # No st.rerun() here; state change will trigger it.
            elif selected_session_id == "-- Select or Start New --" and st.session_state.current_conversation_id is not None:
                # User explicitly selected to start a new conversation from the dropdown
                st.session_state.current_conversation_id = None
                st.session_state.messages = []
                # No st.rerun() here; state change will trigger it.

        except Exception as e:
            st.sidebar.error(f"Error loading conversations: {e}")
            st.session_state.current_conversation_id = None # Clear current if error
            st.session_state.messages = [] # Clear messages

    # --- Display Options ---
    st.sidebar.subheader("Display Options")
    
    # Initialize session state for display options if not already set
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = False
    if "show_author" not in st.session_state:
        st.session_state.show_author = True
    if "auto_collapse_tool_details" not in st.session_state:
        st.session_state.auto_collapse_tool_details = True

    st.sidebar.checkbox("Enable Streaming", value=st.session_state.streaming_enabled, key="streaming_enabled")
    st.sidebar.checkbox("Show Author", value=st.session_state.show_author, key="show_author")
    st.sidebar.checkbox("Auto-collapse Tool Details", value=st.session_state.auto_collapse_tool_details, key="auto_collapse_tool_details")
    # The value is automatically stored in st.session_state.auto_collapse_tool_details by the widget.

    # --- Debug Info (Optional) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Info")
    st.sidebar.write(f"Current Backend: {st.session_state.get('selected_backend', 'ADK')}")
    st.sidebar.write(f"Current User ID: {st.session_state.get('user_id', 'default_user')}")
    st.sidebar.write(f"Current Session ID: {st.session_state.get('current_conversation_id', 'None')}")