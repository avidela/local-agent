"""
WebSocket routes for real-time agent communication.
"""

import json
import asyncio
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...database.models import MessageRole
from ...services import agent_service
from ...services.sessions.session_service import session_service

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, session_id: Optional[str] = None):
        """
        Accept a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            connection_id: Unique connection identifier
            session_id: Optional session ID for grouping
        """
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(connection_id)
    
    def disconnect(self, connection_id: str, session_id: Optional[str] = None):
        """
        Remove a WebSocket connection.
        
        Args:
            connection_id: Connection identifier to remove
            session_id: Optional session ID
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].discard(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
    
    async def send_personal_message(self, message: str, connection_id: str):
        """
        Send a message to a specific connection.
        
        Args:
            message: Message to send
            connection_id: Target connection ID
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message)
            except Exception:
                # Connection closed, remove it
                self.disconnect(connection_id)
    
    async def broadcast_to_session(self, message: str, session_id: str):
        """
        Broadcast a message to all connections in a session.
        
        Args:
            message: Message to broadcast
            session_id: Session ID to broadcast to
        """
        if session_id in self.session_connections:
            disconnected = []
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    websocket = self.active_connections[connection_id]
                    try:
                        await websocket.send_text(message)
                    except Exception:
                        disconnected.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected:
                self.disconnect(connection_id, session_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/agents/{agent_id}/chat")
async def websocket_agent_chat(
    websocket: WebSocket,
    agent_id: int,
    connection_id: str,
    session_id: Optional[str] = None,
    user_id: int = 1  # For demo, use admin user
):
    """
    WebSocket endpoint for real-time agent chat.
    
    Args:
        websocket: WebSocket connection
        agent_id: Agent ID to chat with
        connection_id: Unique connection identifier
        session_id: Optional session ID for conversation context
        user_id: User ID (demo default to admin)
    """
    
    await manager.connect(websocket, connection_id, session_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": f"Connected to agent {agent_id}",
            "agent_id": agent_id,
            "session_id": session_id
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type", "message")
                
                if message_type == "message":
                    await handle_chat_message(
                        websocket=websocket,
                        agent_id=agent_id,
                        user_id=user_id,
                        session_id=session_id,
                        message_data=message_data,
                        connection_id=connection_id
                    )
                elif message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id, session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(connection_id, session_id)


async def handle_chat_message(
    websocket: WebSocket,
    agent_id: int,
    user_id: int,
    session_id: Optional[str],
    message_data: dict,
    connection_id: str
):
    """
    Handle a chat message from the client.
    
    Args:
        websocket: WebSocket connection
        agent_id: Agent ID
        user_id: User ID
        session_id: Session ID
        message_data: Message data from client
        connection_id: Connection ID
    """
    
    prompt = message_data.get("content", "").strip()
    if not prompt:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Empty message content"
        }))
        return
    
    # Get database session (need to handle this differently for WebSocket)
    from ...database import async_session_factory
    
    async with async_session_factory() as db:
        try:
            # Get agent
            agent = await agent_service.get_agent(db, agent_id, user_id)
            if not agent:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Agent not found"
                }))
                return
            
            # Create or get session
            if not session_id:
                session = await session_service.create_session(
                    db=db,
                    user_id=user_id,
                    agent_id=agent_id,
                    title=f"WebSocket Chat with {agent.name}",
                    metadata={"connection_type": "websocket"}
                )
                if session:
                    session_id = str(session.id)
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Failed to create session"
                    }))
                    return
            
            # Add user message to session
            await session_service.add_message(
                db=db,
                session_id=session_id,
                role=MessageRole.USER,
                content={"text": prompt},
                metadata={"connection_id": connection_id}
            )
            
            # Send message received confirmation
            await websocket.send_text(json.dumps({
                "type": "message_received",
                "content": prompt,
                "session_id": session_id
            }))
            
            # Send typing indicator
            await websocket.send_text(json.dumps({
                "type": "typing",
                "message": "Agent is thinking..."
            }))
            
            # Stream agent response
            await stream_agent_response_websocket(
                websocket=websocket,
                agent=agent,
                prompt=prompt,
                session_id=session_id,
                db=db
            )
            
        except Exception as e:
            print(f"Chat message error: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to process message: {str(e)}"
            }))


async def stream_agent_response_websocket(
    websocket: WebSocket,
    agent: Any,
    prompt: str,
    session_id: str,
    db: AsyncSession
):
    """
    Stream agent response through WebSocket.
    
    Args:
        websocket: WebSocket connection
        agent: Database Agent record
        prompt: User prompt
        session_id: Session ID
        db: Database session
    """
    
    try:
        # Get streaming result from PydanticAI
        stream_result = await agent_service.stream_agent(
            agent=agent,
            prompt=prompt
        )
        
        if not stream_result:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Failed to start agent stream"
            }))
            return
        
        # Initialize tracking
        total_cost = 0.0
        total_tokens = 0
        response_text = ""
        
        # Send stream start
        await websocket.send_text(json.dumps({
            "type": "stream_start",
            "message": "Agent response starting..."
        }))
        
        # Stream chunks
        async for chunk in stream_result:
            try:
                # Extract chunk content
                if hasattr(chunk, 'delta') and chunk.delta:
                    if hasattr(chunk.delta, 'content') and chunk.delta.content:
                        content = chunk.delta.content
                        response_text += content
                        
                        # Send content chunk
                        await websocket.send_text(json.dumps({
                            "type": "stream_chunk",
                            "content": content
                        }))
                
                # Track usage and cost
                if hasattr(chunk, 'usage') and chunk.usage:
                    if hasattr(chunk.usage, 'total_tokens'):
                        total_tokens = chunk.usage.total_tokens
                    if hasattr(chunk.usage, 'total_cost'):
                        total_cost = float(chunk.usage.total_cost or 0.0)
                    
                    # Send progress update
                    await websocket.send_text(json.dumps({
                        "type": "progress",
                        "tokens": total_tokens,
                        "cost": total_cost
                    }))
                
                # Tool execution tracking
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_name = getattr(tool_call, 'function', {}).get('name', 'unknown')
                        await websocket.send_text(json.dumps({
                            "type": "tool_execution",
                            "tool_name": tool_name,
                            "status": "executing"
                        }))
                        
            except Exception as chunk_error:
                print(f"WebSocket chunk error: {chunk_error}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Chunk processing error: {str(chunk_error)}"
                }))
        
        # Add final assistant message to session
        await session_service.add_message(
            db=db,
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content={"text": response_text},
            metadata={
                "cost": total_cost,
                "tokens": total_tokens,
                "model": agent.model_name,
                "connection_type": "websocket"
            }
        )
        
        # Send stream completion
        await websocket.send_text(json.dumps({
            "type": "stream_complete",
            "content": response_text,
            "cost": total_cost,
            "tokens": total_tokens,
            "session_id": session_id
        }))
        
    except Exception as e:
        print(f"WebSocket streaming error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Streaming error: {str(e)}"
        }))