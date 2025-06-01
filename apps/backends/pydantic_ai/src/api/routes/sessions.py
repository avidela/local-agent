"""
Sessions API routes for conversation management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...services import session_service
from ..schemas import (
    SessionCreate, SessionUpdate, SessionResponse, MessageResponse,
    SessionStats, PaginationParams
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation session.
    
    Args:
        session_data: Session configuration
        db: Database session
        
    Returns:
        Created session
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    session = await session_service.create_session(
        db=db,
        user_id=user_id,
        agent_id=session_data.agent_id,
        title=session_data.title,
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create session"
        )
    
    return session


@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    pagination: PaginationParams = Depends(),
    agent_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List user sessions with pagination.
    
    Args:
        pagination: Pagination parameters
        agent_id: Filter by agent ID
        db: Database session
        
    Returns:
        List of sessions
    """
    
    try:
        # For demo, use user_id=1 (admin user)
        user_id = 1
        
        sessions = await session_service.list_sessions(
            db=db,
            user_id=user_id,
            agent_id=agent_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
        
        return sessions
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get session by ID.
    
    Args:
        session_id: Session UUID
        db: Database session
        
    Returns:
        Session details
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    session = await session_service.get_session(db, session_id, user_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    session_data: SessionUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update session configuration.
    
    Args:
        session_id: Session UUID
        session_data: Updated session data
        db: Database session
        
    Returns:
        Updated session
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    # Convert to dict and filter None values
    updates = {k: v for k, v in session_data.dict().items() if v is not None}
    
    session = await session_service.update_session(
        db=db,
        session_id=session_id,
        user_id=user_id,
        **updates
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )
    
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Archive session.
    
    Args:
        session_id: Session UUID
        db: Database session
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    success = await session_service.delete_session(db, session_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or unauthorized"
        )


@router.get("/{session_id}/messages", response_model=List[MessageResponse])
async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None,
    include_system: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Get messages for a session.
    
    Args:
        session_id: Session UUID
        limit: Maximum number of messages
        include_system: Include system messages
        db: Database session
        
    Returns:
        List of messages
    """
    
    # Verify session access
    user_id = 1
    session = await session_service.get_session(db, session_id, user_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    messages = await session_service.get_messages(
        db=db,
        session_id=session_id,
        limit=limit,
        include_system=include_system,
    )
    
    return messages


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get session history in ModelMessagesTypeAdapter format.
    
    Args:
        session_id: Session UUID
        db: Database session
        
    Returns:
        Messages in ModelMessagesTypeAdapter format
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    history = await session_service.get_session_history(db, session_id, user_id)
    
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return {"messages": history}


@router.get("/stats/global", response_model=SessionStats)
async def get_global_session_stats(db: AsyncSession = Depends(get_db)):
    """
    Get global session statistics.
    
    Args:
        db: Database session
        
    Returns:
        Global session statistics
    """
    
    stats = await session_service.get_session_stats(db)
    return SessionStats(**stats)


@router.get("/stats/user", response_model=SessionStats)
async def get_user_session_stats(db: AsyncSession = Depends(get_db)):
    """
    Get user session statistics.
    
    Args:
        db: Database session
        
    Returns:
        User session statistics
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    stats = await session_service.get_session_stats(db, user_id=user_id)
    return SessionStats(**stats)