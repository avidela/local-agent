import streamlit as st
from typing import Callable, Optional, List, Any

def chat_input_component(on_send: Callable[[str, Optional[List[Any]]], None], disabled: bool = False):
    """
    A reusable chat input component with file upload capabilities.
    """
    # Streamlit's chat_input returns a dict-like object when accept_file is True/multiple
    # The on_submit callback is triggered when Enter is pressed or file is submitted.
    # We pass the entire dict-like object to the on_send callback.
    user_submission = st.chat_input(
        "Type your message here or upload files...",
        on_submit=lambda: on_send(
            st.session_state.chat_input_key.text,
            st.session_state.chat_input_key.files
        ),
        key="chat_input_key",
        disabled=disabled,
        accept_file="multiple", # Allow multiple files
        file_type=None # Allow all file types for now
    )
    # Note: user_submission itself is None on most reruns.
    # The actual data is in st.session_state.chat_input_key after submission.