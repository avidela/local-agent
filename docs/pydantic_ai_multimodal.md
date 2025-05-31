# 🎨 Multimodal Input Support

> **Image, audio, video, and document input handling using official Pydantic AI patterns**

## 🎯 Overview

Pydantic AI supports multimodal inputs including images, audio, video, and documents through official content types like `ImageUrl`, `BinaryContent`, `AudioUrl`, `VideoUrl`, and `DocumentUrl`. This enables rich agent interactions beyond text.

## 🖼️ Image Input Support

### Core Image Handling Service
```python
# src/services/multimodal_service.py
from pydantic_ai import Agent, ImageUrl, BinaryContent
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import httpx
import base64
import mimetypes
from uuid import UUID

from ..models.media import MediaFile
from ..schemas.multimodal import ImageUpload, MediaResponse, MultimodalMessage

class MultimodalService:
    """Service for handling multimodal inputs with Pydantic AI"""
    
    def __init__(self):
        self.supported_image_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'
        }
        self.supported_audio_types = {
            'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/webm'
        }
        self.supported_video_types = {
            'video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/mov'
        }
        self.supported_document_types = {
            'application/pdf', 'text/plain', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
    
    async def process_image_url(self, url: str) -> ImageUrl:
        """Process image from URL using official Pydantic AI ImageUrl"""
        
        # Validate URL accessibility
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(url)
                if response.status_code != 200:
                    raise ValueError(f"Image URL not accessible: {url}")
                
                content_type = response.headers.get('content-type', '')
                if not any(img_type in content_type for img_type in self.supported_image_types):
                    raise ValueError(f"Unsupported image type: {content_type}")
        
        except httpx.RequestError as e:
            raise ValueError(f"Failed to validate image URL: {str(e)}")
        
        return ImageUrl(url=url)
    
    async def process_image_binary(
        self, 
        image_data: bytes, 
        media_type: str
    ) -> BinaryContent:
        """Process binary image data using official Pydantic AI BinaryContent"""
        
        if media_type not in self.supported_image_types:
            raise ValueError(f"Unsupported image type: {media_type}")
        
        return BinaryContent(data=image_data, media_type=media_type)
    
    async def create_image_message(
        self,
        agent: Agent,
        text_prompt: str,
        image_input: Union[str, bytes, Path],
        media_type: Optional[str] = None
    ) -> List[Union[str, ImageUrl, BinaryContent]]:
        """Create multimodal message with image using official patterns"""
        
        message_parts = [text_prompt]
        
        if isinstance(image_input, str):
            # URL input
            image_content = await self.process_image_url(image_input)
            message_parts.append(image_content)
            
        elif isinstance(image_input, bytes):
            # Binary input
            if not media_type:
                raise ValueError("media_type required for binary image data")
            image_content = await self.process_image_binary(image_input, media_type)
            message_parts.append(image_content)
            
        elif isinstance(image_input, Path):
            # File path input
            if not image_input.exists():
                raise ValueError(f"Image file not found: {image_input}")
            
            # Detect media type
            detected_type = mimetypes.guess_type(str(image_input))[0]
            if not detected_type or detected_type not in self.supported_image_types:
                raise ValueError(f"Unsupported image file type: {detected_type}")
            
            image_data = image_input.read_bytes()
            image_content = await self.process_image_binary(image_data, detected_type)
            message_parts.append(image_content)
        
        return message_parts
    
    async def run_image_analysis(
        self,
        agent: Agent,
        text_prompt: str,
        image_input: Union[str, bytes, Path],
        media_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run agent with image input using official patterns"""
        
        try:
            message_parts = await self.create_image_message(
                agent, text_prompt, image_input, media_type
            )
            
            # Run agent with multimodal input
            result = await agent.run(message_parts)
            
            return {
                "success": True,
                "response": result.data,
                "cost": result.cost(),
                "message_count": len(result.all_messages()),
                "input_type": "image"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "input_type": "image"
            }
```

## 🎵 Audio Input Support

### Audio Processing Implementation
```python
# src/services/multimodal_service.py (continued)
from pydantic_ai import AudioUrl

class MultimodalService:
    # ... existing methods ...
    
    async def process_audio_url(self, url: str) -> AudioUrl:
        """Process audio from URL using official Pydantic AI AudioUrl"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(url)
                if response.status_code != 200:
                    raise ValueError(f"Audio URL not accessible: {url}")
                
                content_type = response.headers.get('content-type', '')
                if not any(audio_type in content_type for audio_type in self.supported_audio_types):
                    raise ValueError(f"Unsupported audio type: {content_type}")
        
        except httpx.RequestError as e:
            raise ValueError(f"Failed to validate audio URL: {str(e)}")
        
        return AudioUrl(url=url)
    
    async def process_audio_binary(
        self, 
        audio_data: bytes, 
        media_type: str
    ) -> BinaryContent:
        """Process binary audio data using official Pydantic AI BinaryContent"""
        
        if media_type not in self.supported_audio_types:
            raise ValueError(f"Unsupported audio type: {media_type}")
        
        return BinaryContent(data=audio_data, media_type=media_type)
    
    async def create_audio_message(
        self,
        agent: Agent,
        text_prompt: str,
        audio_input: Union[str, bytes, Path],
        media_type: Optional[str] = None
    ) -> List[Union[str, AudioUrl, BinaryContent]]:
        """Create multimodal message with audio using official patterns"""
        
        message_parts = [text_prompt]
        
        if isinstance(audio_input, str):
            # URL input
            audio_content = await self.process_audio_url(audio_input)
            message_parts.append(audio_content)
            
        elif isinstance(audio_input, bytes):
            # Binary input
            if not media_type:
                raise ValueError("media_type required for binary audio data")
            audio_content = await self.process_audio_binary(audio_input, media_type)
            message_parts.append(audio_content)
            
        elif isinstance(audio_input, Path):
            # File path input
            if not audio_input.exists():
                raise ValueError(f"Audio file not found: {audio_input}")
            
            detected_type = mimetypes.guess_type(str(audio_input))[0]
            if not detected_type or detected_type not in self.supported_audio_types:
                raise ValueError(f"Unsupported audio file type: {detected_type}")
            
            audio_data = audio_input.read_bytes()
            audio_content = await self.process_audio_binary(audio_data, detected_type)
            message_parts.append(audio_content)
        
        return message_parts
```

## 🎬 Video Input Support

### Video Processing Implementation
```python
# src/services/multimodal_service.py (continued)
from pydantic_ai import VideoUrl

class MultimodalService:
    # ... existing methods ...
    
    async def process_video_url(self, url: str) -> VideoUrl:
        """Process video from URL using official Pydantic AI VideoUrl"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(url)
                if response.status_code != 200:
                    raise ValueError(f"Video URL not accessible: {url}")
                
                content_type = response.headers.get('content-type', '')
                if not any(video_type in content_type for video_type in self.supported_video_types):
                    raise ValueError(f"Unsupported video type: {content_type}")
        
        except httpx.RequestError as e:
            raise ValueError(f"Failed to validate video URL: {str(e)}")
        
        return VideoUrl(url=url)
    
    async def process_video_binary(
        self, 
        video_data: bytes, 
        media_type: str
    ) -> BinaryContent:
        """Process binary video data using official Pydantic AI BinaryContent"""
        
        if media_type not in self.supported_video_types:
            raise ValueError(f"Unsupported video type: {media_type}")
        
        return BinaryContent(data=video_data, media_type=media_type)
    
    async def create_video_message(
        self,
        agent: Agent,
        text_prompt: str,
        video_input: Union[str, bytes, Path],
        media_type: Optional[str] = None
    ) -> List[Union[str, VideoUrl, BinaryContent]]:
        """Create multimodal message with video using official patterns"""
        
        message_parts = [text_prompt]
        
        if isinstance(video_input, str):
            # URL input
            video_content = await self.process_video_url(video_input)
            message_parts.append(video_content)
            
        elif isinstance(video_input, bytes):
            # Binary input
            if not media_type:
                raise ValueError("media_type required for binary video data")
            video_content = await self.process_video_binary(video_input, media_type)
            message_parts.append(video_content)
            
        elif isinstance(video_input, Path):
            # File path input
            if not video_input.exists():
                raise ValueError(f"Video file not found: {video_input}")
            
            detected_type = mimetypes.guess_type(str(video_input))[0]
            if not detected_type or detected_type not in self.supported_video_types:
                raise ValueError(f"Unsupported video file type: {detected_type}")
            
            video_data = video_input.read_bytes()
            video_content = await self.process_video_binary(video_data, detected_type)
            message_parts.append(video_content)
        
        return message_parts
```

## 📄 Document Input Support

### Document Processing Implementation
```python
# src/services/multimodal_service.py (continued)
from pydantic_ai import DocumentUrl

class MultimodalService:
    # ... existing methods ...
    
    async def process_document_url(self, url: str) -> DocumentUrl:
        """Process document from URL using official Pydantic AI DocumentUrl"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(url)
                if response.status_code != 200:
                    raise ValueError(f"Document URL not accessible: {url}")
                
                content_type = response.headers.get('content-type', '')
                if not any(doc_type in content_type for doc_type in self.supported_document_types):
                    raise ValueError(f"Unsupported document type: {content_type}")
        
        except httpx.RequestError as e:
            raise ValueError(f"Failed to validate document URL: {str(e)}")
        
        return DocumentUrl(url=url)
    
    async def process_document_binary(
        self, 
        document_data: bytes, 
        media_type: str
    ) -> BinaryContent:
        """Process binary document data using official Pydantic AI BinaryContent"""
        
        if media_type not in self.supported_document_types:
            raise ValueError(f"Unsupported document type: {media_type}")
        
        # Note: When using Gemini models, document content is always sent as binary
        return BinaryContent(data=document_data, media_type=media_type)
    
    async def create_document_message(
        self,
        agent: Agent,
        text_prompt: str,
        document_input: Union[str, bytes, Path],
        media_type: Optional[str] = None
    ) -> List[Union[str, DocumentUrl, BinaryContent]]:
        """Create multimodal message with document using official patterns"""
        
        message_parts = [text_prompt]
        
        if isinstance(document_input, str):
            # URL input
            document_content = await self.process_document_url(document_input)
            message_parts.append(document_content)
            
        elif isinstance(document_input, bytes):
            # Binary input
            if not media_type:
                raise ValueError("media_type required for binary document data")
            document_content = await self.process_document_binary(document_input, media_type)
            message_parts.append(document_content)
            
        elif isinstance(document_input, Path):
            # File path input
            if not document_input.exists():
                raise ValueError(f"Document file not found: {document_input}")
            
            detected_type = mimetypes.guess_type(str(document_input))[0]
            if not detected_type or detected_type not in self.supported_document_types:
                raise ValueError(f"Unsupported document file type: {detected_type}")
            
            document_data = document_input.read_bytes()
            document_content = await self.process_document_binary(document_data, detected_type)
            message_parts.append(document_content)
        
        return message_parts
```

## 🔄 Multimodal Agent Tools

### File Processing Tools with Multimodal Support
```python
# src/tools/multimodal/processors.py
from pydantic_ai import Agent, RunContext
from typing import Union, List, Dict, Any
from pathlib import Path

from ...services.multimodal_service import MultimodalService

class MultimodalTools:
    """Tools for processing multimodal content in agents"""
    
    def __init__(self):
        self.multimodal_service = MultimodalService()
    
    def create_image_analysis_tool(self):
        """Create image analysis tool using official patterns"""
        
        async def analyze_image(
            ctx: RunContext[None],
            image_input: str,
            analysis_prompt: str = "Describe what you see in this image"
        ) -> str:
            """Analyze image content"""
            
            try:
                # Create agent for image analysis
                from ...services.model_service import ModelService
                model_service = ModelService()
                
                # Use a vision-capable model
                model = model_service.get_model(
                    provider="google",  # or "openai" for GPT-4V
                    model_name="gemini-1.5-flash"
                )
                
                analysis_agent = Agent(model)
                
                # Process image and get analysis
                result = await self.multimodal_service.run_image_analysis(
                    analysis_agent, analysis_prompt, image_input
                )
                
                if result["success"]:
                    return f"Image analysis: {result['response']}"
                else:
                    return f"Image analysis failed: {result['error']}"
                    
            except Exception as e:
                return f"Error analyzing image: {str(e)}"
        
        return analyze_image
    
    def create_document_processing_tool(self):
        """Create document processing tool using official patterns"""
        
        async def process_document(
            ctx: RunContext[None],
            document_input: str,
            processing_prompt: str = "Summarize the main content of this document"
        ) -> str:
            """Process document content"""
            
            try:
                from ...services.model_service import ModelService
                model_service = ModelService()
                
                # Use a document-capable model
                model = model_service.get_model(
                    provider="anthropic",  # Claude is good with documents
                    model_name="claude-3-sonnet-20240229"
                )
                
                processing_agent = Agent(model)
                
                # Create document message
                message_parts = await self.multimodal_service.create_document_message(
                    processing_agent, processing_prompt, document_input
                )
                
                # Process document
                result = await processing_agent.run(message_parts)
                
                return f"Document analysis: {result.data}"
                
            except Exception as e:
                return f"Error processing document: {str(e)}"
        
        return process_document
    
    def create_audio_transcription_tool(self):
        """Create audio transcription tool using official patterns"""
        
        async def transcribe_audio(
            ctx: RunContext[None],
            audio_input: str,
            transcription_prompt: str = "Please transcribe this audio"
        ) -> str:
            """Transcribe audio content"""
            
            try:
                from ...services.model_service import ModelService
                model_service = ModelService()
                
                # Use an audio-capable model
                model = model_service.get_model(
                    provider="google",
                    model_name="gemini-1.5-pro"
                )
                
                transcription_agent = Agent(model)
                
                # Create audio message
                message_parts = await self.multimodal_service.create_audio_message(
                    transcription_agent, transcription_prompt, audio_input
                )
                
                # Transcribe audio
                result = await transcription_agent.run(message_parts)
                
                return f"Audio transcription: {result.data}"
                
            except Exception as e:
                return f"Error transcribing audio: {str(e)}"
        
        return transcribe_audio

multimodal_tools = MultimodalTools()
```

## 🌐 Multimodal API Endpoints

### FastAPI Endpoints for Multimodal Upload
```python
# src/api/v1/multimodal.py
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import json

from ...database.session import get_db
from ...services.multimodal_service import MultimodalService
from ...services.session_service import SessionService
from ...services.agent_service import AgentService
from ...services.model_service import ModelService
from ...schemas.multimodal import MultimodalMessageCreate, MultimodalResponse
from ...auth.dependencies import get_current_user_id

router = APIRouter(prefix="/multimodal", tags=["multimodal"])

@router.post("/analyze-image")
async def analyze_image(
    session_id: UUID,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    streaming: bool = Form(False),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Analyze uploaded image with agent"""
    
    multimodal_service = MultimodalService()
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        # Validate image type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read image data
        image_data = await image.read()
        
        # Get session and agent
        session = await session_service._get_session_with_history(session_id, user_id, db)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        agent = await agent_service.get_agent_instance(session.agent_id, db)
        
        # Create multimodal message
        message_parts = await multimodal_service.create_image_message(
            agent, prompt, image_data, image.content_type
        )
        
        if streaming:
            # Streaming response
            async def stream_analysis():
                async with agent.run_stream(message_parts) as result:
                    async for text_chunk in result.stream_text():
                        yield f"data: {json.dumps({'content': text_chunk, 'partial': True})}\n\n"
                    
                    # Final response
                    yield f"data: {json.dumps({'content': '', 'partial': False, 'complete': True})}\n\n"
            
            return StreamingResponse(
                stream_analysis(),
                media_type="text/plain"
            )
        else:
            # Batch response
            result = await agent.run(message_parts)
            
            return {
                "success": True,
                "response": result.data,
                "cost": result.cost(),
                "media_type": "image",
                "filename": image.filename
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/process-document")
async def process_document(
    session_id: UUID,
    prompt: str = Form(...),
    document: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Process uploaded document with agent"""
    
    multimodal_service = MultimodalService()
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        # Validate document type
        supported_types = {
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        if document.content_type not in supported_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported document type: {document.content_type}"
            )
        
        # Read document data
        document_data = await document.read()
        
        # Get session and agent
        session = await session_service._get_session_with_history(session_id, user_id, db)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        agent = await agent_service.get_agent_instance(session.agent_id, db)
        
        # Create multimodal message
        message_parts = await multimodal_service.create_document_message(
            agent, prompt, document_data, document.content_type
        )
        
        # Process document
        result = await agent.run(message_parts)
        
        return {
            "success": True,
            "response": result.data,
            "cost": result.cost(),
            "media_type": "document",
            "filename": document.filename
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/transcribe-audio")
async def transcribe_audio(
    session_id: UUID,
    audio: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """Transcribe uploaded audio with agent"""
    
    multimodal_service = MultimodalService()
    model_service = ModelService()
    agent_service = AgentService(model_service)
    session_service = SessionService(agent_service)
    
    try:
        # Validate audio type
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be audio"
            )
        
        # Read audio data
        audio_data = await audio.read()
        
        # Get session and agent
        session = await session_service._get_session_with_history(session_id, user_id, db)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        agent = await agent_service.get_agent_instance(session.agent_id, db)
        
        # Create multimodal message
        message_parts = await multimodal_service.create_audio_message(
            agent, "Please transcribe this audio", audio_data, audio.content_type
        )
        
        # Transcribe audio
        result = await agent.run(message_parts)
        
        return {
            "success": True,
            "transcription": result.data,
            "cost": result.cost(),
            "media_type": "audio",
            "filename": audio.filename
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/supported-types")
async def get_supported_types():
    """Get supported media types"""
    
    multimodal_service = MultimodalService()
    
    return {
        "image_types": list(multimodal_service.supported_image_types),
        "audio_types": list(multimodal_service.supported_audio_types),
        "video_types": list(multimodal_service.supported_video_types),
        "document_types": list(multimodal_service.supported_document_types)
    }
```

## 📊 Multimodal Schemas

### Pydantic Schemas for Multimodal Content
```python
# src/schemas/multimodal.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Union
from uuid import UUID
from datetime import datetime

class MediaUpload(BaseModel):
    """Base schema for media uploads"""
    
    filename: str
    content_type: str
    size: int
    description: Optional[str] = None

class ImageUpload(MediaUpload):
    """Schema for image uploads"""
    
    @validator('content_type')
    def validate_image_type(cls, v):
        if not v.startswith('image/'):
            raise ValueError('Content type must be an image type')
        return v

class AudioUpload(MediaUpload):
    """Schema for audio uploads"""
    
    @validator('content_type')
    def validate_audio_type(cls, v):
        if not v.startswith('audio/'):
            raise ValueError('Content type must be an audio type')
        return v

class DocumentUpload(MediaUpload):
    """Schema for document uploads"""
    
    @validator('content_type')
    def validate_document_type(cls, v):
        allowed_types = {
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        if v not in allowed_types:
            raise ValueError(f'Unsupported document type: {v}')
        return v

class MultimodalMessageCreate(BaseModel):
    """Schema for creating multimodal messages"""
    
    text_prompt: str = Field(..., min_length=1)
    media_type: Literal["image", "audio", "video", "document"]
    media_url: Optional[str] = None
    streaming: bool = False
    
    @validator('media_url')
    def validate_media_url(cls, v, values):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Media URL must be a valid HTTP/HTTPS URL')
        return v

class MultimodalResponse(BaseModel):
    """Schema for multimodal analysis responses"""
    
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    cost: Optional[float] = None
    media_type: str
    filename: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: datetime = datetime.utcnow()

class MediaFileInfo(BaseModel):
    """Schema for media file information"""
    
    id: UUID
    filename: str
    content_type: str
    size: int
    upload_date: datetime
    user_id: UUID
    session_id: Optional[UUID] = None
    description: Optional[str] = None
    
    class Config:
        from_attributes = True
```

## 🚀 Key Features

**🎨 Official Multimodal API Integration:**
- [`ImageUrl(url=...)`](https://docs.pydantic.ai/api/messages/#pydantic_ai.ImageUrl) for image URLs
- [`BinaryContent(data=..., media_type=...)`](https://docs.pydantic.ai/api/messages/#pydantic_ai.BinaryContent) for binary content
- [`AudioUrl`](https://docs.pydantic.ai/api/messages/#pydantic_ai.AudioUrl), [`VideoUrl`](https://docs.pydantic.ai/api/messages/#pydantic_ai.VideoUrl), [`DocumentUrl`](https://docs.pydantic.ai/api/messages/#pydantic_ai.DocumentUrl) for respective media types

**📁 Comprehensive Media Support:**
- **Images**: JPEG, PNG, GIF, WebP, BMP formats
- **Audio**: MP3, WAV, OGG, M4A, WebM formats  
- **Video**: MP4, WebM, OGG, AVI, MOV formats
- **Documents**: PDF, TXT, DOC, DOCX formats

**🔄 Flexible Input Methods:**
- URL-based media input with validation
- Binary data upload with type checking
- File path processing with automatic type detection
- Streaming responses for real-time analysis

**