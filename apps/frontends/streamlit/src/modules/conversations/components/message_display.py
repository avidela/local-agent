import streamlit as st
import json # Import json module
import json # Already imported, but ensuring it's there
import base64 # Import base64 for decoding file data
from typing import List, Optional, Dict, Any
from src.shared.models.common_models import Message, MessagePart

def render_message(message: Message, expand_tool_details: bool = False):
    """
    Renders a single message, handling multimodal content, tool calls, and author display.
    """
    with st.chat_message(message.author):
        for part in message.parts:
            if part.text:
                st.markdown(part.text)
            elif part.image_url:
                st.image(part.image_url)
            elif part.file_data:
                # Assuming file_data is a dict with "mime_type" and "data" (base64 encoded)
                mime_type = part.file_data.get("mime_type")
                data = part.file_data.get("data")
                if data and mime_type:
                    import base64 # Import base64 here as it's only needed for file data
                    if "image" in mime_type:
                        st.image(base64.b64decode(data), caption=part.file_data.get("display_name", "Image"))
                    elif "video" in mime_type:
                        st.video(base64.b64decode(data), format=mime_type, start_time=0)
                    elif "pdf" in mime_type:
                        st.download_button(
                            label=f"Download {part.file_data.get('display_name', 'PDF Document')}",
                            data=base64.b64decode(data),
                            file_name=part.file_data.get("display_name", "document.pdf"),
                            mime=mime_type
                        )
                    else:
                        st.warning(f"Unsupported file type for display: {mime_type}")
                else:
                    st.warning("Malformed file_data received.")

        if message.is_tool_call:
            tool_details = f"Tool: `{message.tool_name}`"
            if message.tool_args:
                tool_details += f"\nArgs: `{json.dumps(message.tool_args, indent=2)}`"
            
            if expand_tool_details:
                st.expander("Tool Call Details").code(tool_details, language="json")
            else:
                st.info(f"Tool Call: `{message.tool_name}` (click to expand)")
                with st.expander("Tool Call Details"):
                    st.code(tool_details, language="json")

        if message.is_tool_response and message.tool_response_content:
            if expand_tool_details:
                st.expander("Tool Response").code(message.tool_response_content, language="json")
            else:
                st.success(f"Tool Response (click to expand)")
                with st.expander("Tool Response"):
                    st.code(message.tool_response_content, language="json")

def message_display(messages: List[Message], show_author: bool = True, auto_collapse_tool_details: bool = True):
    """
    Displays a list of messages in a chat-like interface.
    """
    for message in messages:
        # Streamlit's chat_message handles author display, but we can customize
        # if show_author is False, we might just render content without the bubble
        render_message(message, expand_tool_details=not auto_collapse_tool_details)