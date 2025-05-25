from typing import List, Optional, Dict, Any, Union, Generator
from src.shared.models.common_models import Conversation, Message, MessagePart
from src.services.adk_backend.adk_api_client import adk_backend_client
from src.services.adk_backend.adk_models import SessionADK, EventADK, MessagePartADK
from src.services.backend_adapters.adk_backend_adapter import ADKBackendAdapter # Import from new location
from src.services.backend_client_protocol import BackendClient # Import BackendClient protocol

# --- Backend Selector Logic ---

def get_active_backend_client() -> BackendClient:
    """
    Returns the currently selected backend client based on Streamlit session state.
    In a real application, this might read from a dropdown in the sidebar.
    """
    import streamlit as st # Import st locally as it's only used here
    selected_backend_type = st.session_state.get("selected_backend", "ADK")

    if selected_backend_type == "ADK":
        return ADKBackendAdapter()
    # elif selected_backend_type == "LANGGRAPH":
    #     return LangGraphBackendAdapter()
    # elif selected_backend_type == "PYDANTIC_AI":
    #     return PydanticAIBackendAdapter()
    else:
        raise ValueError(f"Unknown backend type: {selected_backend_type}")

# Global instance to be used by modules/services
active_backend_client = get_active_backend_client()