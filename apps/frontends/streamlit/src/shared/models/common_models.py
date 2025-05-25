from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class MessagePart:
    text: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None # Added for video content
    file_data: Optional[Dict[str, Any]] = None # {"mime_type": "application/pdf", "data": "base64_encoded_pdf", "display_name": "my_doc.pdf"}
    function_call: Optional[Dict[str, Any]] = None # For tool calls
    function_response: Optional[Dict[str, Any]] = None # For tool responses

@dataclass
class Message:
    author: str # "user", "agent", "tool"
    parts: List[MessagePart] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    is_tool_response: bool = False
    tool_response_content: Optional[str] = None
    # Add other metadata like invocation_id, branch, etc. if needed for UI display

@dataclass
class Conversation:
    id: str
    user_id: str
    app_name: str
    messages: List[Message] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    last_update_time: float = field(default_factory=lambda: datetime.now().timestamp())

# You can add more common models here as your application grows,
# e.g., User, AgentConfig, etc.