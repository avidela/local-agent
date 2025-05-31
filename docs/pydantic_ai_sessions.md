# 💬 Session & Message Management

> **Conversation history and message handling using official Pydantic AI message APIs**

## 🎯 Overview

The Session Management system provides persistent conversation history, streaming support, and message handling using official Pydantic AI message APIs. This enables multi-turn conversations with proper context preservation.

## 💾 Session Service Implementation

### Core Session Service with Message History
```python
# src/services/session_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID, uuid4
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python

from ..models.session import Session
from ..models.event import Event
from ..schemas.sessions import SessionCreate, SessionResponse, MessageCreate, EventResponse
from .agent_service import AgentService

class SessionService:
    """Service for managing sessions with proper message history using official Pydantic AI API"""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
    
    async def create_session(
        self,
        session_data: SessionCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> SessionResponse:
        """Create a new session with message history support"""
        
        session = Session(
            agent_id=session_data.agent_id,
            user_id=user_id,
            external_session_id=session_data.external_session_id or str(uuid4()),
            state=session_data.initial_state or {},
            message_history=[]  # Initialize empty message history
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        return SessionResponse.from_orm(session)
    
    async def send_message(
        self,
        session_id: UUID,
        message: MessageCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> AsyncGenerator[EventResponse, None]:
        """Send message and get agent response with proper conversation history"""
        
        # Get session with message history
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session:
            raise ValueError("Session not found")
        
        # Get agent instance
        agent = await self.agent_service.get_agent_instance(session.agent_id, db)
        
        # Prepare message history from previous conversations
        message_history = None
        if session.message_history:
            try:
                # Convert stored JSON back to message objects using official API
                message_history = ModelMessagesTypeAdapter.validate_python(session.message_history)
            except Exception:
                # If message history is corrupted, start fresh
                message_history = None
        
        # Execute agent with conversation history
        if message.streaming:
            async for event in self._execute_streaming_with_history(
                agent, message.content, message_history, session, db
            ):
                yield event
        else:
            event = await self._execute_batch_with_history(
                agent, message.content, message_history, session, db
            )
            yield event
    
    async def _execute_batch_with_history(
        self,
        agent: Agent,
        user_message: str,
        message_history: Optional[List],
        session: Session,
        db: AsyncSession
    ) -> EventResponse:
        """Execute agent with message history in batch mode using official API"""
        
        try:
            # Run agent with message history for conversation continuity
            if message_history:
                result = await agent.run(user_message, message_history=message_history)
            else:
                result = await agent.run(user_message)
            
            # Store user message event
            await self._store_user_event(session.id, user_message, db)
            
            # Store agent response event with usage info
            agent_event = await self._store_agent_event(
                session.id, 
                result.data, 
                result.cost(),
                len(result.all_messages()),
                db
            )
            
            # Update session message history with new messages using official API
            await self._update_session_history(session, result.all_messages(), db)
            
            return EventResponse.from_orm(agent_event)
            
        except Exception as e:
            # Store error event
            error_event = await self._store_error_event(session.id, str(e), db)
            return EventResponse.from_orm(error_event)
    
    async def _execute_streaming_with_history(
        self,
        agent: Agent,
        user_message: str,
        message_history: Optional[List],
        session: Session,
        db: AsyncSession
    ) -> AsyncGenerator[EventResponse, None]:
        """Execute agent with message history in streaming mode using official API"""
        
        try:
            # Store user message first
            await self._store_user_event(session.id, user_message, db)
            
            # Start streaming with message history using official API
            if message_history:
                stream_result = agent.run_stream(user_message, message_history=message_history)
            else:
                stream_result = agent.run_stream(user_message)
            
            accumulated_response = ""
            
            async with stream_result as result:
                # Stream partial responses using official stream_text() method
                async for text_chunk in result.stream_text():
                    accumulated_response += text_chunk
                    
                    # Create partial event
                    partial_event = Event(
                        session_id=session.id,
                        author="agent",
                        content=text_chunk,
                        metadata={"partial": True},
                        partial=True,
                        turn_complete=False,
                        sequence_number=await self._get_next_sequence_number(session.id, db)
                    )
                    
                    db.add(partial_event)
                    await db.commit()
                    await db.refresh(partial_event)
                    
                    yield EventResponse.from_orm(partial_event)
                
                # Store final complete response with usage info
                final_event = await self._store_agent_event(
                    session.id,
                    accumulated_response,
                    result.cost(),
                    len(result.all_messages()),
                    db
                )
                
                # Update session message history using official all_messages() method
                await self._update_session_history(session, result.all_messages(), db)
                
                yield EventResponse.from_orm(final_event)
                
        except Exception as e:
            error_event = await self._store_error_event(session.id, str(e), db)
            yield EventResponse.from_orm(error_event)
    
    async def _update_session_history(
        self,
        session: Session,
        all_messages: List,
        db: AsyncSession
    ):
        """Update session with latest message history for conversation continuity using official API"""
        
        try:
            # Convert messages to JSON for storage using official to_jsonable_python
            session.message_history = to_jsonable_python(all_messages)
            await db.commit()
        except Exception:
            # If serialization fails, keep existing history
            pass
    
    async def get_conversation_history(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session or not session.message_history:
            return []
        
        try:
            # Convert stored message history back to readable format
            messages = ModelMessagesTypeAdapter.validate_python(session.message_history)
            return [
                {
                    "type": self._determine_message_type(msg),
                    "content": self._extract_message_content(msg),
                    "timestamp": getattr(msg, "timestamp", None)
                }
                for msg in messages
            ]
        except Exception:
            return []
    
    async def get_session_events(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0
    ) -> List[EventResponse]:
        """Get events for a session with pagination"""
        
        from sqlalchemy import select, desc
        
        # Verify session ownership
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session:
            raise ValueError("Session not found")
        
        # Get events with pagination
        query = (
            select(Event)
            .where(Event.session_id == session_id)
            .order_by(desc(Event.sequence_number))
            .limit(limit)
            .offset(offset)
        )
        
        result = await db.execute(query)
        events = result.scalars().all()
        
        return [EventResponse.from_orm(event) for event in events]
    
    async def update_session_state(
        self,
        session_id: UUID,
        state_updates: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> SessionResponse:
        """Update session state for stateful agents"""
        
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session:
            raise ValueError("Session not found")
        
        # Update state
        session.state.update(state_updates)
        await db.commit()
        await db.refresh(session)
        
        return SessionResponse.from_orm(session)
    
    async def clear_conversation_history(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> bool:
        """Clear conversation history while keeping session"""
        
        session = await self._get_session_with_history(session_id, user_id, db)
        if not session:
            raise ValueError("Session not found")
        
        # Clear message history
        session.message_history = []
        await db.commit()
        
        return True
    
    def _determine_message_type(self, message) -> str:
        """Determine message type from Pydantic AI message object"""
        
        # Check if message has parts to determine type
        if hasattr(message, "parts"):
            for part in message.parts:
                if "User" in str(type(part)):
                    return "user"
                elif "System" in str(type(part)):
                    return "system"
        
        # Check if it's a model response
        if hasattr(message, "usage") or "ModelResponse" in str(type(message)):
            return "assistant"
        
        return "unknown"
    
    def _extract_message_content(self, message) -> str:
        """Extract content from Pydantic AI message object"""
        
        if hasattr(message, "parts"):
            # Extract content from message parts
            content_parts = []
            for part in message.parts:
                if hasattr(part, "content"):
                    content_parts.append(str(part.content))
            return " ".join(content_parts)
        
        # For model responses, convert to string
        return str(message)
    
    async def _store_user_event(self, session_id: UUID, content: str, db: AsyncSession) -> Event:
        """Store user message event"""
        event = Event(
            session_id=session_id,
            author="user",
            content=content,
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _store_agent_event(
        self, 
        session_id: UUID, 
        content: str, 
        cost: float,
        message_count: int,
        db: AsyncSession
    ) -> Event:
        """Store agent response event with usage information"""
        event = Event(
            session_id=session_id,
            author="agent",
            content=content,
            metadata={
                "cost": cost,
                "message_count": message_count,
                "partial": False
            },
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _store_error_event(self, session_id: UUID, error_message: str, db: AsyncSession) -> Event:
        """Store error event"""
        event = Event(
            session_id=session_id,
            author="system",
            content="",
            error_code="EXECUTION_ERROR",
            error_message=error_message,
            partial=False,
            turn_complete=True,
            sequence_number=await self._get_next_sequence_number(session_id, db)
        )
        
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return event
    
    async def _get_session_with_history(self, session_id: UUID, user_id: UUID, db: AsyncSession) -> Optional[Session]:
        """Get session with message history"""
        from sqlalchemy import select
        
        result = await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id
            )
        )
        return result.scalar_one_or_none()
    
    async def _get_next_sequence_number(self, session_id: UUID, db: AsyncSession) -> int:
        """Get next sequence number for event"""
        from sqlalchemy import select, func
        
        result = await db.execute(
            select(func.max(Event.sequence_number))
            .where(Event.session_id == session_id)
        )
        last_seq = result.scalar() or 0
        return last_seq + 1
```

## 📊 Session Management Schemas

### Session and Message Schemas
```python
# src/schemas/sessions.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
from uuid import UUID
from datetime import datetime

class SessionCreate(BaseModel):
    """Schema for creating new sessions"""
    
    agent_id: UUID
    external_session_id: Optional[str] = None
    initial_state: Optional[Dict[str, Any]] = {}

class SessionUpdate(BaseModel):
    """Schema for updating sessions"""
    
    state: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    """Schema for session responses"""
    
    id: UUID
    agent_id: UUID
    user_id: UUID
    external_session_id: Optional[str]
    message_history: List[Dict[str, Any]]
    state: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    """Schema for creating messages"""
    
    content: str = Field(..., min_length=1, max_length=10000)
    streaming: bool = False
    metadata: Dict[str, Any] = {}

class EventResponse(BaseModel):
    """Schema for event responses"""
    
    id: UUID
    session_id: UUID
    author: Literal["user", "agent", "system"]
    content: str
    metadata: Dict[str, Any]
    partial: bool
    turn_complete: bool
    sequence_number: int
    error_code: Optional[str]
    error_message: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True

class ConversationHistory(BaseModel):
    """Schema for conversation history"""
    
    session_id: UUID
    messages: List[Dict[str, Any]]
    total_cost: float
    total_messages: int

class SessionStats(BaseModel):
    """Schema for session statistics"""
    
    session_id: UUID
    total_events: int
    user_messages: int
    agent_responses: int
    errors: int
    total_cost: float
    average_response_time: Optional[float]
    created_at: datetime
    last_activity: datetime
```

## 🔄 Message History Utilities

### Message History Utilities
```python
# src/utils/message_history.py
from typing import List, Dict, Any, Optional
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python, from_json
import json

class MessageHistoryUtils:
    """Utilities for handling Pydantic AI message history"""
    
    @staticmethod
    def serialize_messages(messages: List) -> List[Dict[str, Any]]:
        """Serialize Pydantic AI messages to JSON-compatible format"""
        try:
            return to_jsonable_python(messages)
        except Exception as e:
            raise ValueError(f"Failed to serialize messages: {str(e)}")
    
    @staticmethod
    def deserialize_messages(json_messages: List[Dict[str, Any]]) -> List:
        """Deserialize JSON back to Pydantic AI message objects"""
        try:
            return ModelMessagesTypeAdapter.validate_python(json_messages)
        except Exception as e:
            raise ValueError(f"Failed to deserialize messages: {str(e)}")
    
    @staticmethod
    def extract_conversation_summary(messages: List) -> Dict[str, Any]:
        """Extract summary information from message history"""
        
        summary = {
            "total_messages": len(messages),
            "user_messages": 0,
            "assistant_messages": 0,
            "system_messages": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        for message in messages:
            # Determine message type
            if hasattr(message, "parts"):
                for part in message.parts:
                    if "User" in str(type(part)):
                        summary["user_messages"] += 1
                    elif "System" in str(type(part)):
                        summary["system_messages"] += 1
            elif hasattr(message, "usage"):
                summary["assistant_messages"] += 1
                
                # Extract usage information if available
                usage = getattr(message, "usage", {})
                if hasattr(usage, "total_tokens"):
                    summary["total_tokens"] += usage.total_tokens
        
        return summary
    
    @staticmethod
    def truncate_history(
        messages: List, 
        max_messages: int = 100,
        preserve_system: bool = True
    ) -> List:
        """Truncate message history while preserving important messages"""
        
        if len(messages) <= max_messages:
            return messages
        
        truncated = []
        
        # Preserve system messages if requested
        if preserve_system:
            system_messages = []
            other_messages = []
            
            for message in messages:
                if hasattr(message, "parts"):
                    has_system = any("System" in str(type(part)) for part in message.parts)
                    if has_system:
                        system_messages.append(message)
                    else:
                        other_messages.append(message)
                else:
                    other_messages.append(message)
            
            # Keep all system messages and most recent other messages
            available_slots = max_messages - len(system_messages)
            truncated = system_messages + other_messages[-available_slots:]
        else:
            # Simply keep the most recent messages
            truncated = messages[-max_messages:]
        
        return truncated
    
    @staticmethod
    def calculate_context_length(messages: List) -> int:
        """Estimate context length of message history"""
        
        total_length = 0
        
        for message in messages:
            # Simple estimation based on string length
            # In practice, you'd use a proper tokenizer
            message_str = str(message)
            total_length += len(message_str.split())  # Rough word count
        
        return total_length
    
    @staticmethod
    def filter_messages_by_type(
        messages: List, 
        message_types: List[str]
    ) -> List:
        """Filter messages by type (user, assistant, system)"""
        
        filtered = []
        
        for message in messages:
            message_type = MessageHistoryUtils._determine_message_type(message)
            if message_type in message_types:
                filtered.append(message)
        
        return filtered
    
    @staticmethod
    def _determine_message_type(message) -> str:
        """Determine message type from Pydantic AI message object"""
        
        if hasattr(message, "parts"):
            for part in message.parts:
                if "User" in str(type(part)):
                    return "user"
                elif "System" in str(type(part)):
                    return "system"
        
        if hasattr(message, "usage") or "ModelResponse" in str(type(message)):
            return "assistant"
        
        return "unknown"

# Global utilities instance
message_utils = MessageHistoryUtils()
```

## 📈 Session Analytics

### Session Analytics Service
```python
# src/services/session_analytics.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc

from ..models.session import Session
from ..models.event import Event
from ..schemas.sessions import SessionStats

class SessionAnalyticsService:
    """Service for session analytics and insights"""
    
    async def get_session_stats(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> SessionStats:
        """Get statistics for a specific session"""
        
        # Verify session ownership
        session_query = select(Session).where(
            Session.id == session_id,
            Session.user_id == user_id
        )
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            raise ValueError("Session not found")
        
        # Get event statistics
        event_stats_query = select(
            func.count(Event.id).label("total_events"),
            func.sum(func.case((Event.author == "user", 1), else_=0)).label("user_messages"),
            func.sum(func.case((Event.author == "agent", 1), else_=0)).label("agent_responses"),
            func.sum(func.case((Event.error_code.isnot(None), 1), else_=0)).label("errors"),
            func.max(Event.timestamp).label("last_activity")
        ).where(Event.session_id == session_id)
        
        stats_result = await db.execute(event_stats_query)
        stats = stats_result.first()
        
        # Calculate total cost from event metadata
        cost_query = select(Event.metadata).where(
            Event.session_id == session_id,
            Event.author == "agent",
            Event.metadata.op("->>")('"cost"').isnot(None)
        )
        cost_result = await db.execute(cost_query)
        
        total_cost = 0.0
        for row in cost_result:
            metadata = row[0] or {}
            if "cost" in metadata:
                total_cost += float(metadata["cost"])
        
        return SessionStats(
            session_id=session_id,
            total_events=stats.total_events or 0,
            user_messages=stats.user_messages or 0,
            agent_responses=stats.agent_responses or 0,
            errors=stats.errors or 0,
            total_cost=total_cost,
            average_response_time=None,  # Would need timing data
            created_at=session.created_at,
            last_activity=stats.last_activity or session.created_at
        )
    
    async def get_user_session_summary(
        self,
        user_id: UUID,
        db: AsyncSession,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get session summary for a user over specified period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get session counts and activity
        session_query = select(
            func.count(Session.id).label("total_sessions"),
            func.count(func.case((Session.created_at >= cutoff_date, 1))).label("recent_sessions"),
            func.max(Session.updated_at).label("last_activity")
        ).where(Session.user_id == user_id)
        
        session_result = await db.execute(session_query)
        session_stats = session_result.first()
        
        # Get event counts for user's sessions
        event_query = select(
            func.count(Event.id).label("total_events"),
            func.sum(func.case((Event.author == "user", 1), else_=0)).label("user_messages"),
            func.sum(func.case((Event.author == "agent", 1), else_=0)).label("agent_responses")
        ).select_from(
            Event.join(Session, Event.session_id == Session.id)
        ).where(
            Session.user_id == user_id,
            Event.timestamp >= cutoff_date
        )
        
        event_result = await db.execute(event_query)
        event_stats = event_result.first()
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_sessions": session_stats.total_sessions or 0,
            "recent_sessions": session_stats.recent_sessions or 0,
            "total_events": event_stats.total_events or 0,
            "user_messages": event_stats.user_messages or 0,
            "agent_responses": event_stats.agent_responses or 0,
            "last_activity": session_stats.last_activity
        }
    
    async def get_popular_agents(
        self,
        db: AsyncSession,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most popular agents by session count"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            Session.agent_id,
            func.count(Session.id).label("session_count"),
            func.count(func.distinct(Session.user_id)).label("unique_users")
        ).where(
            Session.created_at >= cutoff_date
        ).group_by(
            Session.agent_id
        ).order_by(
            desc("session_count")
        ).limit(limit)
        
        result = await db.execute(query)
        
        return [
            {
                "agent_id": row.agent_id,
                "session_count": row.session_count,
                "unique_users": row.unique_users
            }
            for row in result
        ]

analytics_service = SessionAnalyticsService()
```

## 🚀 Key Features

**💬 Official Message API Integration:**
- [`result.all_messages()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for complete conversation history
- [`result.new_messages()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for current turn messages
- [`ModelMessagesTypeAdapter`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for JSON serialization
- [`to_jsonable_python()`](https://docs.pydantic.ai/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) for storage

**🔄 Conversation Continuity:**
- [`agent.run(message, message_history=history)`](https://docs.pydantic.ai/agents/#pydantic_ai.Agent.run) for context preservation
- [`agent.run_stream(message, message_history=history)`](https://docs.pydantic.ai/agents/#pydantic_ai.Agent.run_stream) for streaming with context
- Persistent message history storage in PostgreSQL JSON fields
- Automatic conversation state management

**📊 Streaming Support:**
- [`result.stream_text()`](https://docs.pydantic.ai/agents/#pydantic_ai.StreamedRunResult.stream_text) for real-time responses
- Partial event tracking during streaming
- Complete message history update after stream completion
- Error handling for interrupted streams

**💰 Cost & Usage Tracking:**
- [`result.cost()`](https://docs.pydantic.ai/api/agents/#pydantic_ai.RunResult.cost) for accurate cost tracking
- Message count and token usage analytics
- Session-level cost aggregation
- User activity and usage summaries

**🔧 Advanced Features:**
- Message history truncation and management
- Session state persistence for stateful agents
- Event-driven architecture for real-time updates
- Comprehensive analytics and insights

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*