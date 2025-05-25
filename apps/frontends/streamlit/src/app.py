import streamlit as st
from src.core.components.sidebar_controls import sidebar_controls
from src.modules.conversations.pages.conversations_page import conversations_page

st.set_page_config(
    page_title="ADK Monorepo UI",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables if they don't exist
if "selected_backend" not in st.session_state:
    st.session_state.selected_backend = "ADK" # Default backend
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user" # Default user ID
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = False
if "show_author" not in st.session_state:
    st.session_state.show_author = True
if "auto_collapse_tool_details" not in st.session_state:
    st.session_state.auto_collapse_tool_details = True
# Remove default selected_adk_app initialization here.
# It will be set by sidebar_controls after fetching available apps.

# Render the sidebar controls
with st.sidebar:
    sidebar_controls()

# Main content area
st.markdown("# ADK Monorepo UI")

# For now, we only have one main page: conversations
# In a multi-page app, you'd use st.sidebar.radio or similar for navigation
# and then call the appropriate page function.
conversations_page()