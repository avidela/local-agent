"""
Agents API routes using official PydanticAI patterns.
"""

import json
import asyncio
from typing import Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...database.models import MessageRole
from ...services import agent_service, model_provider_service
from ...services.sessions.session_service import session_service
from ...observability import trace_agent_run
from ..schemas import (
    AgentCreate, AgentUpdate, AgentResponse, AgentRunRequest, AgentRunResponse,
    AgentStats, PaginationParams, ModelInfo, ProviderStatus
)

router = APIRouter()


@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new dynamic agent.
    
    Args:
        agent_data: Agent configuration
        db: Database session
        
    Returns:
        Created agent
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    try:
        agent = await agent_service.create_agent(
            db=db,
            user_id=user_id,
            name=agent_data.name,
            description=agent_data.description,
            system_prompt=agent_data.system_prompt,
            model_provider=agent_data.model_provider,
            model_name=agent_data.model_name,
            tools=agent_data.tools,
            output_type=agent_data.output_type,
            temperature=agent_data.temperature,
            max_tokens=agent_data.max_tokens,
            retries=agent_data.retries,
            is_public=agent_data.is_public,
        )
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create agent"
            )
        
        return agent
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[AgentResponse])
async def list_agents(
    pagination: PaginationParams = Depends(),
    include_public: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    List agents with pagination.
    
    Args:
        pagination: Pagination parameters
        include_public: Include public agents
        db: Database session
        
    Returns:
        List of agents
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    agents = await agent_service.list_agents(
        db=db,
        user_id=user_id,
        include_public=include_public,
        skip=pagination.skip,
        limit=pagination.limit,
    )
    
    return agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get agent by ID.
    
    Args:
        agent_id: Agent ID
        db: Database session
        
    Returns:
        Agent details
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    agent = await agent_service.get_agent(db, agent_id, user_id)
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    agent_data: AgentUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update agent configuration.
    
    Args:
        agent_id: Agent ID
        agent_data: Updated agent data
        db: Database session
        
    Returns:
        Updated agent
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    # Convert to dict and filter None values
    updates = {k: v for k, v in agent_data.dict().items() if v is not None}
    
    agent = await agent_service.update_agent(
        db=db,
        agent_id=agent_id,
        user_id=user_id,
        **updates
    )
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or unauthorized"
        )
    
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete agent (soft delete).
    
    Args:
        agent_id: Agent ID
        db: Database session
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    success = await agent_service.delete_agent(db, agent_id, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or unauthorized"
        )


def _is_message_complete(content: str, buffer: str) -> bool:
    """
    Determine if a streaming content chunk represents a complete message unit.
    
    Args:
        content: The current content chunk
        buffer: The accumulated content buffer
        
    Returns:
        True if the message appears complete, False if partial
    """
    
    # Simple heuristics for message completion detection
    # These can be adjusted based on your specific streaming patterns
    
    # Check for sentence endings
    sentence_endings = ['.', '!', '?', '\n\n']
    if any(content.rstrip().endswith(ending) for ending in sentence_endings):
        return True
    
    # Check for paragraph breaks (double newlines)
    if '\n\n' in content:
        return True
    
    # Check for code block endings
    if content.rstrip().endswith('```'):
        return True
    
    # Check for list item completion (ends with newline after bullet point content)
    if content.rstrip().endswith('\n') and buffer.strip() and any(buffer.strip().startswith(marker) for marker in ['- ', '* ', '1. ', '2. ', '3. ']):
        return True
    
    # Check for structured data endings (JSON, etc.)
    if content.rstrip().endswith('}') or content.rstrip().endswith(']'):
        return True
    
    # If content is very short (likely incomplete)
    if len(content.strip()) < 3:
        return False
    
    # Default to partial for safety
    return False


async def stream_agent_response(
    agent_service: Any,
    agent: Any,
    run_request: AgentRunRequest,
    session_id: str,
    db: AsyncSession,
    **run_kwargs
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for agent response.
    
    Args:
        agent_service: Agent service instance
        agent: Database Agent record
        run_request: Run request configuration
        session_id: Session ID for context
        db: Database session
        **run_kwargs: Additional run arguments
        
    Yields:
        SSE-formatted chunks
    """
    
    try:
        # Start streaming
        yield f"data: {json.dumps({'type': 'start', 'content': 'Starting agent response...'})}\n\n"
        
        # Get the streaming context manager from PydanticAI
        stream_context = await agent_service.stream_agent(
            agent=agent,
            prompt=run_request.prompt,
            image_urls=run_request.image_urls,
            document_urls=run_request.document_urls,
            audio_urls=run_request.audio_urls,
            video_urls=run_request.video_urls,
            db=db,
            user_id=run_kwargs.get('user_id'),
            **{k: v for k, v in run_kwargs.items() if k not in ['user_id']}
        )
        
        if not stream_context:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to start agent stream'})}\n\n"
            return
        
        # Initialize cost and token tracking
        total_cost = 0.0
        total_tokens = 0
        response_text = ""
        
        # Use the async context manager to get the stream
        async with stream_context as stream_result:
            # Track message processing for PydanticAI streaming
            current_tool = None
            content_buffer = ""
            last_complete_message = ""  # Track the last complete message for database storage
            
            # Process each message in the stream
            async for message in stream_result.stream():
                try:
                    # Debug: print message type for troubleshooting
                    print(f"Stream message type: {type(message)}")
                    print(f"Stream message content: {repr(message)}")
                    
                    # Handle string messages (direct content)
                    if isinstance(message, str):
                        content = message
                        if content.strip():  # Only send non-empty content
                            # For PydanticAI, each chunk often contains the full text so far
                            # We'll use the content as is for streaming, but track the latest complete version
                            content_buffer = content  # Replace buffer with latest content
                            
                            # Determine if this chunk appears to be a complete message unit
                            is_complete = _is_message_complete(content, content_buffer)
                            
                            # Send content chunk with partial/complete indicator
                            content_event = {
                                'type': 'content',
                                'content': content,
                                'partial': not is_complete,
                                'buffer_size': len(content_buffer)
                            }
                            yield f"data: {json.dumps(content_event)}\n\n"
                            
                            # If this appears to be a complete message unit, update last complete message
                            if is_complete:
                                last_complete_message = content_buffer.strip()
                                boundary_event = {
                                    'type': 'message_boundary',
                                    'content': last_complete_message,
                                    'length': len(last_complete_message)
                                }
                                yield f"data: {json.dumps(boundary_event)}\n\n"
                    
                    # Handle structured message objects
                    elif hasattr(message, 'parts') and message.parts:
                        for part in message.parts:
                            print(f"Message part type: {type(part)}")
                            
                            # Handle text content parts (primary content)
                            if hasattr(part, 'text') and part.text and part.text.strip():
                                content = part.text
                                content_buffer = content  # Replace with latest content
                                
                                is_complete = _is_message_complete(content, content_buffer)
                                
                                content_event = {
                                    'type': 'content',
                                    'content': content,
                                    'partial': not is_complete,
                                    'buffer_size': len(content_buffer)
                                }
                                yield f"data: {json.dumps(content_event)}\n\n"
                                
                                if is_complete:
                                    last_complete_message = content_buffer.strip()
                                    boundary_event = {
                                        'type': 'message_boundary',
                                        'content': last_complete_message,
                                        'length': len(last_complete_message)
                                    }
                                    yield f"data: {json.dumps(boundary_event)}\n\n"
                            
                            # Handle legacy content attribute for backwards compatibility
                            elif hasattr(part, 'content') and isinstance(part.content, str) and part.content.strip():
                                content = part.content
                                content_buffer = content  # Replace with latest content
                                
                                is_complete = _is_message_complete(content, content_buffer)
                                
                                content_event = {
                                    'type': 'content',
                                    'content': content,
                                    'partial': not is_complete,
                                    'buffer_size': len(content_buffer)
                                }
                                yield f"data: {json.dumps(content_event)}\n\n"
                                
                                if is_complete:
                                    last_complete_message = content_buffer.strip()
                                    boundary_event = {
                                        'type': 'message_boundary',
                                        'content': last_complete_message,
                                        'length': len(last_complete_message)
                                    }
                                    yield f"data: {json.dumps(boundary_event)}\n\n"
                            
                            # Handle Gemini-specific parts gracefully (skip empty/unsupported parts)
                            elif hasattr(part, 'function_call') and part.function_call:
                                # Handle function calls (tool calls)
                                tool_name = getattr(part.function_call, 'name', 'unknown')
                                current_tool = tool_name
                                tool_event = {
                                    'type': 'tool',
                                    'tool_name': tool_name,
                                    'status': 'executing',
                                    'partial': False,
                                    'args': getattr(part.function_call, 'args', {})
                                }
                                yield f"data: {json.dumps(tool_event)}\n\n"
                            
                            elif hasattr(part, 'function_response') and part.function_response:
                                # Handle function responses (tool responses)
                                if current_tool:
                                    tool_result = str(getattr(part.function_response, 'response', ''))
                                    tool_response_event = {
                                        'type': 'tool',
                                        'tool_name': current_tool,
                                        'status': 'completed',
                                        'result': tool_result,
                                        'partial': False
                                    }
                                    yield f"data: {json.dumps(tool_response_event)}\n\n"
                                    current_tool = None
                            
                            # Skip other Gemini-specific parts (thoughts, video_metadata, etc.)
                            # These are informational but not needed for streaming content
                            else:
                                # Debug: Log skipped parts without causing errors
                                part_attrs = [attr for attr in dir(part) if not attr.startswith('_')]
                                print(f"Skipping unsupported part with attributes: {part_attrs}")
                            
                    
                    # Handle direct message content attributes
                    elif hasattr(message, 'content'):
                        if isinstance(message.content, str):
                            content = message.content
                            if content.strip():
                                response_text += content
                                content_buffer = content  # Replace with latest content
                                
                                is_complete = _is_message_complete(content, content_buffer)
                                
                                content_event = {
                                    'type': 'content',
                                    'content': content,
                                    'partial': not is_complete,
                                    'buffer_size': len(content_buffer)
                                }
                                yield f"data: {json.dumps(content_event)}\n\n"
                                
                                if is_complete:
                                    last_complete_message = content_buffer.strip()
                                    boundary_event = {
                                        'type': 'message_boundary',
                                        'content': last_complete_message,
                                        'length': len(last_complete_message)
                                    }
                                    yield f"data: {json.dumps(boundary_event)}\n\n"
                        elif hasattr(message.content, 'text'):
                            content = message.content.text
                            if content and content.strip():
                                content_buffer = content  # Replace with latest content
                                
                                is_complete = _is_message_complete(content, content_buffer)
                                
                                content_event = {
                                    'type': 'content',
                                    'content': content,
                                    'partial': not is_complete,
                                    'buffer_size': len(content_buffer)
                                }
                                yield f"data: {json.dumps(content_event)}\n\n"
                                
                                if is_complete:
                                    last_complete_message = content_buffer.strip()
                                    boundary_event = {
                                        'type': 'message_boundary',
                                        'content': last_complete_message,
                                        'length': len(last_complete_message)
                                    }
                                    yield f"data: {json.dumps(boundary_event)}\n\n"
                    
                    # Handle tool calls at message level (always complete events)
                    elif hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = getattr(tool_call, 'name', getattr(tool_call, 'function', {}).get('name', 'unknown'))
                            tool_event = {
                                'type': 'tool',
                                'tool_name': tool_name,
                                'status': 'executing',
                                'partial': False
                            }
                            yield f"data: {json.dumps(tool_event)}\n\n"
                        
                except Exception as chunk_error:
                    print(f"Error processing stream message: {chunk_error}")
                    print(f"Message: {repr(message)}")
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Stream processing error: {str(chunk_error)}'})}\n\n"
            
            # Handle any remaining content in buffer as final complete message
            if content_buffer.strip():
                last_complete_message = content_buffer.strip()
                final_boundary_event = {
                    'type': 'message_boundary',
                    'content': last_complete_message,
                    'length': len(last_complete_message),
                    'final': True
                }
                yield f"data: {json.dumps(final_boundary_event)}\n\n"
            
            # Use the last complete message for database storage (clean, no duplicates)
            final_message_for_db = last_complete_message if last_complete_message else content_buffer.strip()
            
            # Get final result data from the run result
            try:
                if hasattr(stream_result, 'cost'):
                    total_cost = float(stream_result.cost() or 0.0)
                if hasattr(stream_result, 'usage'):
                    usage = stream_result.usage()
                    if hasattr(usage, 'total_tokens'):
                        total_tokens = usage.total_tokens
                    elif hasattr(usage, 'request_tokens') and hasattr(usage, 'response_tokens'):
                        total_tokens = usage.request_tokens + usage.response_tokens
                        
                # If no content was streamed, get the final data
                if not final_message_for_db and hasattr(stream_result, 'data'):
                    final_message_for_db = str(stream_result.data)
                    final_content_event = {
                        'type': 'content',
                        'content': final_message_for_db,
                        'partial': False,
                        'final': True
                    }
                    yield f"data: {json.dumps(final_content_event)}\n\n"
                    
            except Exception as result_error:
                print(f"Error extracting result data: {result_error}")
        
        # Add final assistant message to session using clean message (no duplicates)
        await session_service.add_message(
            db=db,
            session_id=str(session_id),
            role=MessageRole.ASSISTANT,
            content={"text": final_message_for_db},
            metadata={
                "cost": total_cost,
                "tokens": total_tokens,
                "model": agent.model_name
            }
        )
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'content': final_message_for_db, 'cost': total_cost, 'tokens': total_tokens})}\n\n"
        
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming error: {str(e)}'})}\n\n"


@router.post("/{agent_id}/run")
async def run_agent(
    agent_id: int,
    run_request: AgentRunRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Run agent with official PydanticAI patterns.
    
    Args:
        agent_id: Agent ID
        run_request: Run configuration
        db: Database session
        
    Returns:
        Agent run result
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    # Get agent
    agent = await agent_service.get_agent(db, agent_id, user_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    # Trace the agent run
    with trace_agent_run(agent.name, run_request.prompt):
        try:
            # Create or get session
            session_id = run_request.session_id
            if not session_id:
                # Create new session
                session = await session_service.create_session(
                    db=db,
                    user_id=user_id,
                    agent_id=agent_id,
                    title=f"Chat with {agent.name}",
                    metadata=run_request.metadata
                )
                if not session:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create session"
                    )
                session_id = session.session_id
            else:
                # Verify session exists and user has access
                session = await session_service.get_session(db, session_id, user_id)
                if not session:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Session not found or unauthorized"
                    )
            
            # Add user message to session
            user_message = await session_service.add_message(
                db=db,
                session_id=str(session_id),
                role=MessageRole.USER,
                content={"text": run_request.prompt},
                metadata={"timestamp": "now"}
            )
            
            # Prepare run arguments
            run_kwargs = {}
            
            if run_request.message_history:
                run_kwargs["message_history"] = run_request.message_history
            elif session_id:
                # Get session history for context, but don't include in run_kwargs
                # Let PydanticAI handle conversation state naturally
                pass
            
            if run_request.temperature_override is not None:
                run_kwargs["temperature"] = run_request.temperature_override
            
            if run_request.max_tokens_override is not None:
                run_kwargs["max_tokens"] = run_request.max_tokens_override
            
            # Run agent using official PydanticAI API
            if run_request.stream:
                # Return SSE streaming response
                return StreamingResponse(
                    stream_agent_response(
                        agent_service=agent_service,
                        agent=agent,
                        run_request=run_request,
                        session_id=str(session_id),
                        db=db,
                        user_id=user_id,  # Add user_id for file access
                        **run_kwargs
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    }
                )
            else:
                # Regular response
                result = await agent_service.run_agent(
                    agent=agent,
                    prompt=run_request.prompt,
                    image_urls=run_request.image_urls,
                    document_urls=run_request.document_urls,
                    audio_urls=run_request.audio_urls,
                    video_urls=run_request.video_urls,
                    db=db,
                    user_id=user_id,
                    **run_kwargs
                )
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to run agent"
                )
            
            # Extract information from PydanticAI result
            # Extract output from PydanticAI result
            output_text = str(result.data) if hasattr(result, 'data') else str(result)
            
            # Extract cost from usage (Google Vertex AI often returns 0 cost for free tier)
            cost = 0.0
            if hasattr(result, 'usage') and result.usage:
                usage_obj = result.usage
                if hasattr(usage_obj, 'total_cost'):
                    cost = float(usage_obj.total_cost)
                elif hasattr(usage_obj, 'cost'):
                    cost = float(usage_obj.cost)
            print(f"DEBUG - Extracted cost: {cost}")
            
            # Extract token usage from individual messages (more accurate)
            total_usage_tokens = 0
            total_request_tokens = 0
            total_response_tokens = 0
            
            # Process all messages from the run and persist tool calls
            if hasattr(result, 'all_messages') and result.all_messages:
                all_run_messages = result.all_messages()
                print(f"DEBUG - Found {len(all_run_messages)} messages in run")
                
                for msg in all_run_messages:
                    print(f"DEBUG - Message type: {type(msg)}")
                    
                    # Extract token usage from message if available
                    msg_tokens = 0
                    msg_request_tokens = 0
                    msg_response_tokens = 0
                    
                    if hasattr(msg, 'usage') and msg.usage:
                        usage = msg.usage
                        msg_tokens = getattr(usage, 'total_tokens', 0)
                        msg_request_tokens = getattr(usage, 'request_tokens', 0)
                        msg_response_tokens = getattr(usage, 'response_tokens', 0)
                        
                        # Accumulate totals
                        total_usage_tokens += msg_tokens
                        total_request_tokens += msg_request_tokens
                        total_response_tokens += msg_response_tokens
                        
                        print(f"DEBUG - Message token usage: total={msg_tokens}, request={msg_request_tokens}, response={msg_response_tokens}")
                    
                    # Skip system prompts and user messages (already added)
                    if hasattr(msg, 'parts') and msg.parts:
                        for part in msg.parts:
                            print(f"DEBUG - Part type: {type(part)}")
                            
                            # Handle tool calls (from ModelResponse)
                            if hasattr(part, 'tool_name') and hasattr(part, 'args'):
                                tool_call_data = {
                                    "tool_name": part.tool_name,
                                    "args": part.args,
                                    "tool_call_id": getattr(part, 'tool_call_id', None)
                                }
                                print(f"DEBUG - Saving tool call: {tool_call_data}")
                                
                                # Add tool call message with token usage
                                await session_service.add_message(
                                    db=db,
                                    session_id=session_id,
                                    role=MessageRole.ASSISTANT,
                                    content={"tool_call": tool_call_data},
                                    tool_calls=tool_call_data,
                                    tokens=msg_request_tokens,
                                    metadata={"model": f"{agent.model_provider}:{agent.model_name}", "type": "tool_call"}
                                )
                            
                            # Handle tool responses (from ModelRequest)
                            elif hasattr(part, 'tool_name') and hasattr(part, 'content') and hasattr(part, 'tool_call_id'):
                                tool_response_data = {
                                    "tool_name": part.tool_name,
                                    "content": part.content,
                                    "tool_call_id": part.tool_call_id
                                }
                                print(f"DEBUG - Saving tool response: {tool_response_data}")
                                
                                # Add tool response message
                                await session_service.add_message(
                                    db=db,
                                    session_id=session_id,
                                    role=MessageRole.ASSISTANT,
                                    content={"tool_response": tool_response_data},
                                    tool_response=tool_response_data,
                                    tokens=0,  # Tool response doesn't consume tokens
                                    metadata={"model": f"{agent.model_provider}:{agent.model_name}", "type": "tool_response"}
                                )
            
            # Add final assistant message with the text response
            assistant_message = await session_service.add_message(
                db=db,
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content={"text": output_text},
                cost=cost,
                tokens=total_response_tokens,
                metadata={"model": f"{agent.model_provider}:{agent.model_name}", "type": "final_response"}
            )
            
            # Update session with final token counts
            await session_service.update_session(
                db=db,
                session_id=session_id,
                user_id=user_id,
                total_cost=cost,
                total_tokens=total_usage_tokens,
                request_tokens=total_request_tokens,
                response_tokens=total_response_tokens
            )
            
            print(f"DEBUG - Final token counts: total={total_usage_tokens}, request={total_request_tokens}, response={total_response_tokens}")
            
            # Get all messages for response
            all_messages = await session_service.get_messages(db, session_id, limit=50)
            messages_data = session_service.messages_to_model_format(all_messages)
            
            return AgentRunResponse(
                output=output_text,
                session_id=session_id,
                cost=cost,
                tokens=total_usage_tokens,
                request_tokens=total_request_tokens,
                response_tokens=total_response_tokens,
                messages=messages_data,
                metadata=run_request.metadata,
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed: {str(e)}"
            )


@router.get("/{agent_id}/stats", response_model=dict)
async def get_agent_stats(
    agent_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get agent usage statistics.
    
    Args:
        agent_id: Agent ID
        db: Database session
        
    Returns:
        Agent statistics
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    agent = await agent_service.get_agent(db, agent_id, user_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    return {
        "agent_id": agent.id,
        "name": agent.name,
        "usage_count": agent.usage_count,
        "model": f"{agent.model_provider}:{agent.model_name}",
        "is_public": agent.is_public,
        "created_at": agent.created_at,
    }


@router.get("/stats/global", response_model=AgentStats)
async def get_global_agent_stats(db: AsyncSession = Depends(get_db)):
    """
    Get global agent statistics.
    
    Args:
        db: Database session
        
    Returns:
        Global agent statistics
    """
    
    stats = await agent_service.get_agent_stats(db)
    return AgentStats(**stats)


@router.get("/models/available", response_model=List[ModelInfo])
async def get_available_models():
    """
    Get list of available models.
    
    Returns:
        List of available models with capabilities
    """
    
    models = model_provider_service.list_available_models()
    
    model_list = []
    for model_name, model_info in models.items():
        model_list.append(ModelInfo(
            name=model_name,
            provider=model_info["provider"],
            max_tokens=model_info["max_tokens"],
            supports_tools=model_info["supports_tools"],
            supports_streaming=model_info["supports_streaming"],
            supports_multimodal=model_info["supports_multimodal"],
            context_window=model_info["context_window"],
        ))
    
    return model_list


@router.get("/providers/status", response_model=List[ProviderStatus])
async def get_provider_status():
    """
    Get status of all model providers.
    
    Returns:
        List of provider statuses
    """
    
    providers = []
    
    for provider in model_provider_service.get_supported_providers():
        models = model_provider_service.get_models_by_provider(provider)
        
        model_infos = []
        for model_name, model_info in models.items():
            model_infos.append(ModelInfo(
                name=model_name,
                provider=model_info["provider"],
                max_tokens=model_info["max_tokens"],
                supports_tools=model_info["supports_tools"],
                supports_streaming=model_info["supports_streaming"],
                supports_multimodal=model_info["supports_multimodal"],
                context_window=model_info["context_window"],
            ))
        
        providers.append(ProviderStatus(
            provider=provider,
            available=len(models) > 0,
            models=model_infos,
        ))
    
    return providers