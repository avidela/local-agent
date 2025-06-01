"""
Session service for conversation management with ModelMessagesTypeAdapter support.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import Session, Message, Agent, User, SessionStatus, MessageRole


class SessionService:
    """Service for managing conversation sessions with persistent message history."""
    
    async def create_session(
        self,
        db: AsyncSession,
        user_id: int,
        agent_id: int,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Session]:
        """
        Create a new conversation session.
        
        Args:
            db: Database session
            user_id: User ID
            agent_id: Agent ID
            title: Session title
            metadata: Additional metadata
            
        Returns:
            Created Session instance or None if failed
        """
        
        # Verify agent exists and user has access
        agent_query = select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.is_active == True,
                (Agent.owner_id == user_id) | (Agent.is_public == True)
            )
        )
        agent_result = await db.execute(agent_query)
        agent = agent_result.scalar_one_or_none()
        
        if not agent:
            return None
        
        # Create session
        session = Session(
            session_id=str(uuid4()),
            title=title or f"Chat with {agent.name}",
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        return session
    
    async def get_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[Session]:
        """
        Get session by ID with optional user access validation.
        
        Args:
            db: Database session
            session_id: Session UUID
            user_id: User ID for access validation
            
        Returns:
            Session instance or None if not found/accessible
        """
        
        query = select(Session).where(Session.session_id == session_id)
        
        if user_id is not None:
            query = query.where(Session.user_id == user_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def list_sessions(
        self,
        db: AsyncSession,
        user_id: int,
        agent_id: Optional[int] = None,
        status: Optional[SessionStatus] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Session]:
        """
        List user sessions with optional filtering.
        
        Args:
            db: Database session
            user_id: User ID
            agent_id: Filter by agent ID
            status: Filter by session status
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Session instances
        """
        
        query = select(Session).where(Session.user_id == user_id)
        
        if agent_id:
            query = query.where(Session.agent_id == agent_id)
        
        if status:
            query = query.where(Session.status == status)
        
        # Apply pagination and ordering
        query = query.offset(skip).limit(limit).order_by(Session.created_at.desc())
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def update_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: int,
        **updates
    ) -> Optional[Session]:
        """
        Update session configuration.
        
        Args:
            db: Database session
            session_id: Session UUID
            user_id: User ID for ownership validation
            **updates: Fields to update
            
        Returns:
            Updated Session instance or None if not found/unauthorized
        """
        
        session = await self.get_session(db, session_id, user_id)
        if not session:
            return None
        
        # Update allowed fields
        allowed_fields = {'title', 'status', 'metadata'}
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(session, field):
                setattr(session, field, value)
        
        await db.commit()
        await db.refresh(session)
        
        return session
    
    async def delete_session(self, db: AsyncSession, session_id: str, user_id: int) -> bool:
        """
        Delete session and all associated messages.
        
        Args:
            db: Database session
            session_id: Session UUID
            user_id: User ID for ownership validation
            
        Returns:
            True if deleted, False if not found/unauthorized
        """
        
        session = await self.get_session(db, session_id, user_id)
        if not session:
            return False
        
        # Mark as archived instead of hard delete
        session.status = SessionStatus.ARCHIVED
        
        await db.commit()
        return True
    
    async def add_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: MessageRole,
        content: Dict[str, Any],
        attachments: Optional[List[str]] = None,
        cost: float = 0.0,
        tokens: int = 0,
        tool_calls: Optional[Dict[str, Any]] = None,
        tool_response: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Message]:
        """
        Add a message to the session with ModelMessagesTypeAdapter compatible format.
        
        Args:
            db: Database session
            session_id: Session UUID
            role: Message role (system, user, assistant, tool)
            content: Message content in ModelMessagesTypeAdapter format
            attachments: List of file IDs
            cost: Cost for this message
            tokens: Token count for this message
            tool_calls: Tool calls data
            tool_response: Tool response data
            metadata: Additional metadata
            
        Returns:
            Created Message instance or None if failed
        """
        
        # Get session by database ID
        session_query = select(Session).where(Session.session_id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            return None
        
        # Create message
        message = Message(
            session_id=session.id,
            role=role,
            content=content,
            attachments=attachments or [],
            cost=cost,
            tokens=tokens,
            tool_calls=tool_calls,
            tool_response=tool_response,
            metadata=metadata or {},
        )
        
        db.add(message)
        
        # Update session totals
        session.total_cost += cost
        session.total_tokens += tokens
        
        if role == MessageRole.USER:
            session.request_tokens += tokens
        elif role == MessageRole.ASSISTANT:
            session.response_tokens += tokens
        
        await db.commit()
        await db.refresh(message)
        
        return message
    
    async def get_messages(
        self,
        db: AsyncSession,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Message]:
        """
        Get messages for a session in chronological order.
        
        Args:
            db: Database session
            session_id: Session UUID
            limit: Maximum number of messages to return
            include_system: Whether to include system messages
            
        Returns:
            List of Message instances
        """
        
        # Get session by database ID
        session_query = select(Session).where(Session.session_id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            return []
        
        query = select(Message).where(Message.session_id == session.id)
        
        if not include_system:
            query = query.where(Message.role != MessageRole.SYSTEM)
        
        query = query.order_by(Message.created_at.asc())
        
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    def messages_to_model_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert database messages to ModelMessagesTypeAdapter format.
        
        Args:
            messages: List of Message instances
            
        Returns:
            List of messages in ModelMessagesTypeAdapter format
        """
        
        model_messages = []
        
        for message in messages:
            # Convert to ModelMessagesTypeAdapter format
            model_message = {
                "role": message.role.value if hasattr(message.role, 'value') else message.role,
                "content": message.content,
                "timestamp": str(message.created_at) if message.created_at else str(datetime.now()),
            }
            
            # Add tool calls if present
            if message.tool_calls:
                model_message["tool_calls"] = message.tool_calls
            
            # Add tool response if present
            if message.tool_response:
                model_message["tool_response"] = message.tool_response
            
            # Add attachments if present
            if message.attachments:
                model_message["attachments"] = message.attachments
            
            model_messages.append(model_message)
        
        return model_messages
    
    async def get_session_history(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get complete session history in ModelMessagesTypeAdapter format.
        
        Args:
            db: Database session
            session_id: Session UUID
            user_id: User ID for access validation
            
        Returns:
            List of messages in ModelMessagesTypeAdapter format or None if not accessible
        """
        
        session = await self.get_session(db, session_id, user_id)
        if not session:
            return None
        
        messages = await self.get_messages(db, session_id)
        return self.messages_to_model_format(messages)
    
    async def update_session_costs(
        self,
        db: AsyncSession,
        session_id: str,
        additional_cost: float,
        additional_tokens: int,
        request_tokens: int = 0,
        response_tokens: int = 0,
    ) -> bool:
        """
        Update session cost and token tracking.
        
        Args:
            db: Database session
            session_id: Session UUID
            additional_cost: Additional cost to add
            additional_tokens: Additional tokens to add
            request_tokens: Additional request tokens
            response_tokens: Additional response tokens
            
        Returns:
            True if updated successfully
        """
        
        session_query = select(Session).where(Session.session_id == session_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()
        
        if not session:
            return False
        
        session.total_cost += additional_cost
        session.total_tokens += additional_tokens
        session.request_tokens += request_tokens
        session.response_tokens += response_tokens
        
        await db.commit()
        return True
    
    async def get_session_stats(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        agent_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            db: Database session
            user_id: Filter by user ID
            agent_id: Filter by agent ID
            
        Returns:
            Dictionary with session statistics
        """
        
        query = select(Session)
        
        if user_id:
            query = query.where(Session.user_id == user_id)
        
        if agent_id:
            query = query.where(Session.agent_id == agent_id)
        
        result = await db.execute(query)
        sessions = list(result.scalars().all())
        
        stats = {
            "total_sessions": len(sessions),
            "active_sessions": sum(1 for s in sessions if s.status == SessionStatus.ACTIVE),
            "completed_sessions": sum(1 for s in sessions if s.status == SessionStatus.COMPLETED),
            "total_cost": sum(s.total_cost for s in sessions),
            "total_tokens": sum(s.total_tokens for s in sessions),
            "total_request_tokens": sum(s.request_tokens for s in sessions),
            "total_response_tokens": sum(s.response_tokens for s in sessions),
            "average_cost_per_session": 0.0,
            "average_tokens_per_session": 0.0,
        }
        
        if stats["total_sessions"] > 0:
            stats["average_cost_per_session"] = stats["total_cost"] / stats["total_sessions"]
            stats["average_tokens_per_session"] = stats["total_tokens"] / stats["total_sessions"]
        
        return stats


# Global session service instance
session_service = SessionService()