from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union

# These models are derived from your ADK backend's OpenAPI (Swagger) spec.
# Only include the ones you actually need for the Streamlit frontend.

class MessagePartADK(BaseModel):
    text: Optional[Union[str, bool]] = None
    inlineData: Optional[Dict[str, Any]] = Field(None, alias="inlineData")
    functionCall: Optional[Dict[str, Any]] = Field(None, alias="functionCall")
    functionResponse: Optional[Dict[str, Any]] = Field(None, alias="functionResponse")

    @model_validator(mode='before')
    def convert_bool_to_str(cls, values):
        if isinstance(values, dict) and 'text' in values:
            if isinstance(values['text'], bool):
                values['text'] = str(values['text'])
        return values

class ContentInputADK(BaseModel):
    parts: List[MessagePartADK] = Field(default_factory=list)
    role: Optional[str] = None

class AgentRunRequestADK(BaseModel):
    appName: str
    userId: str
    sessionId: str
    newMessage: ContentInputADK
    streaming: bool = False

class ContentOutputADK(BaseModel):
    parts: List[MessagePartADK] = Field(default_factory=list)
    role: Optional[str] = None

class EventADK(BaseModel):
    author: str
    content: Optional[ContentOutputADK] = None
    invocationId: str = ""
    # Add other fields from Event schema if needed for display or logic
    actions: Optional[Dict[str, Any]] = None # For tool calls, etc.
    timestamp: float # Assuming timestamp is always present and float

class SessionADK(BaseModel):
    id: str
    appName: str
    userId: str
    state: Dict[str, Any] = Field(default_factory=dict)
    events: List[EventADK] = Field(default_factory=list)
    lastUpdateTime: float = 0.0

# Add other ADK-specific models as needed, e.g., for eval sets, artifacts, etc.