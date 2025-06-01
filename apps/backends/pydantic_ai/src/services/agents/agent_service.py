"""
Agent service for dynamic agent creation and management using official PydanticAI patterns.
"""

from typing import Any, Dict, List, Optional, Union
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import settings
from ...database import Agent, User, ModelProvider
from ...api.schemas import ToolConfig
from ..models import model_provider_service
from ..tools.tool_service import get_tool_service


class AgentService:
    """Service for managing dynamic PydanticAI agents."""
    
    def __init__(self):
        """Initialize agent service."""
        self._agent_cache: Dict[int, Any] = {}  # Cache for instantiated agents
    
    async def create_agent(
        self,
        db: AsyncSession,
        user_id: int,
        name: str,
        description: Optional[str],
        system_prompt: str,
        model_provider: ModelProvider,
        model_name: str,
        tools: Optional[List[ToolConfig]] = None,
        output_type: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retries: int = 2,
        is_public: bool = False,
    ) -> Optional[Agent]:
        """
        Create a new dynamic agent in the database.
        
        Args:
            db: Database session
            user_id: ID of the user creating the agent
            name: Agent name
            description: Agent description
            system_prompt: System prompt for the agent
            model_provider: Model provider (OpenAI, Anthropic, Google)
            model_name: Name of the specific model
            tools: List of structured tool configurations
            output_type: Expected output type name
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            retries: Number of retries on failure
            is_public: Whether agent is publicly accessible
            
        Returns:
            Created Agent instance or None if failed
        """
        
        # Validate model availability
        provider_value = model_provider.value if hasattr(model_provider, 'value') else model_provider
        full_model_name = f"{provider_value}:{model_name}"
        if not model_provider_service.is_model_available(full_model_name):
            raise ValueError(f"Model {full_model_name} is not available")
        
        # Convert ToolConfig objects to dictionaries for database storage
        tool_dicts = []
        if tools:
            for tool in tools:
                if isinstance(tool, ToolConfig):
                    tool_dicts.append(tool.model_dump())
                else:
                    # Handle legacy string tools by converting to ToolConfig format
                    tool_dicts.append({
                        "name": str(tool),
                        "enabled": True,
                        "plain": False,
                        "config": {}
                    })
        
        # Create database record
        agent = Agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_dicts,
            output_type=output_type,
            retries=retries,
            is_public=is_public,
            owner_id=user_id,
        )
        
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        return agent
    
    async def get_agent(self, db: AsyncSession, agent_id: int, user_id: Optional[int] = None) -> Optional[Agent]:
        """
        Get agent by ID with optional user access validation.
        
        Args:
            db: Database session
            agent_id: Agent ID
            user_id: User ID for access validation
            
        Returns:
            Agent instance or None if not found/accessible
        """
        
        query = select(Agent).where(Agent.id == agent_id)
        
        # Add access control
        if user_id is not None:
            query = query.where(
                and_(
                    Agent.is_active == True,
                    (Agent.owner_id == user_id) | (Agent.is_public == True)
                )
            )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def list_agents(
        self,
        db: AsyncSession,
        user_id: Optional[int] = None,
        include_public: bool = True,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Agent]:
        """
        List agents with optional filtering.
        
        Args:
            db: Database session
            user_id: User ID to filter by ownership
            include_public: Whether to include public agents
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of Agent instances
        """
        
        query = select(Agent).where(Agent.is_active == True)
        
        # Apply access filters
        if user_id is not None:
            if include_public:
                query = query.where(
                    (Agent.owner_id == user_id) | (Agent.is_public == True)
                )
            else:
                query = query.where(Agent.owner_id == user_id)
        elif include_public:
            query = query.where(Agent.is_public == True)
        
        # Apply pagination
        query = query.offset(skip).limit(limit).order_by(Agent.created_at.desc())
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def update_agent(
        self,
        db: AsyncSession,
        agent_id: int,
        user_id: int,
        **updates
    ) -> Optional[Agent]:
        """
        Update agent configuration.
        
        Args:
            db: Database session
            agent_id: Agent ID
            user_id: User ID for ownership validation
            **updates: Fields to update
            
        Returns:
            Updated Agent instance or None if not found/unauthorized
        """
        
        # Get agent with ownership validation
        agent = await self.get_agent(db, agent_id, user_id)
        if not agent or agent.owner_id != user_id:
            return None
        
        # Update allowed fields
        allowed_fields = {
            'name', 'description', 'system_prompt', 'model_provider', 
            'model_name', 'temperature', 'max_tokens', 'tools', 
            'output_type', 'retries', 'is_public'
        }
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(agent, field):
                setattr(agent, field, value)
        
        # Clear cache if agent was cached
        if agent.id in self._agent_cache:
            del self._agent_cache[agent.id]
        
        await db.commit()
        await db.refresh(agent)
        
        return agent
    
    async def delete_agent(self, db: AsyncSession, agent_id: int, user_id: int) -> bool:
        """
        Soft delete agent (mark as inactive).
        
        Args:
            db: Database session
            agent_id: Agent ID
            user_id: User ID for ownership validation
            
        Returns:
            True if deleted, False if not found/unauthorized
        """
        
        agent = await self.get_agent(db, agent_id, user_id)
        if not agent or agent.owner_id != user_id:
            return False
        
        agent.is_active = False
        
        # Clear cache if agent was cached
        if agent.id in self._agent_cache:
            del self._agent_cache[agent.id]
        
        await db.commit()
        return True
    
    def get_instantiated_agent(self, agent: Agent) -> Optional[Any]:
        """
        Get instantiated PydanticAI Agent from database record.
        
        Args:
            agent: Database Agent record
            
        Returns:
            Instantiated PydanticAI Agent or None if failed
        """
        
        # Check cache first
        if agent.id in self._agent_cache:
            return self._agent_cache[agent.id]
        
        # Create full model name
        provider_value = agent.model_provider.value if hasattr(agent.model_provider, 'value') else agent.model_provider
        full_model_name = f"{provider_value}:{agent.model_name}"
        
        # Get output type if specified
        output_type = None
        if agent.output_type:
            # For now, use string - in a real implementation you'd resolve this
            # to actual Pydantic models based on your type registry
            output_type = agent.output_type
        
        # Extract enabled tool configurations
        enabled_tools = []
        tool_configs = {}
        if agent.tools:
            for tool_dict in agent.tools:
                if isinstance(tool_dict, dict) and tool_dict.get("enabled", True):
                    tool_name = tool_dict["name"]
                    enabled_tools.append(tool_name)
                    tool_configs[tool_name] = tool_dict.get("config", {})
                else:
                    # Handle legacy string format
                    enabled_tools.append(str(tool_dict))
        
        # Get tool service and create tool functions
        tool_service = get_tool_service()
        tool_functions = {}
        
        if enabled_tools:
            try:
                # Create base config for tools (can include workspace path, etc.)
                base_config = {
                    "base_path": getattr(settings, "REPO_ROOT", "/app"),
                    "agent_id": agent.id
                }
                
                # Create tool functions for PydanticAI
                tool_functions = tool_service.create_tool_functions_for_agent(
                    tool_names=enabled_tools,
                    base_config=base_config
                )
                
                # Update tool configs with specific configurations
                for tool_name, config in tool_configs.items():
                    if tool_name in tool_functions:
                        # You could inject specific config here if needed
                        pass
                        
            except Exception as e:
                print(f"Warning: Failed to create tool functions: {e}")
                tool_functions = {}
        
        # Create agent using model provider service with actual tool functions
        pydantic_agent = model_provider_service.create_agent(
            model_name=full_model_name,
            system_prompt=agent.system_prompt,
            tools=list(tool_functions.values()),  # Pass actual callable functions
            output_type=output_type,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            retries=agent.retries,
        )
        
        # Cache the agent
        if pydantic_agent:
            self._agent_cache[agent.id] = pydantic_agent
        
        return pydantic_agent
    
    async def run_agent(
        self,
        agent: Agent,
        prompt: str,
        message_history: Optional[List[Any]] = None,
        deps: Optional[Any] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Run agent with the given prompt using official PydanticAI API.
        
        Args:
            agent: Database Agent record
            prompt: User prompt
            message_history: Previous conversation messages
            deps: Dependencies for agent execution
            **kwargs: Additional arguments for agent.run()
            
        Returns:
            Agent result or None if failed
        """
        
        pydantic_agent = self.get_instantiated_agent(agent)
        if not pydantic_agent:
            return None
        
        try:
            # Use official PydanticAI agent.run() API
            run_kwargs: Dict[str, Any] = {
                "user_prompt": prompt,
            }
            
            if message_history:
                run_kwargs["message_history"] = message_history
            
            if deps:
                run_kwargs["deps"] = deps
            
            # Add any additional kwargs
            run_kwargs.update(kwargs)
            
            # Import here to avoid dependency issues during development
            result = await pydantic_agent.run(**run_kwargs)
            
            # Update usage count
            agent.usage_count += 1
            
            return result
            
        except Exception as e:
            print(f"Error running agent {agent.id}: {e}")
            return None
    
    async def stream_agent(
        self,
        agent: Agent,
        prompt: str,
        message_history: Optional[List[Any]] = None,
        deps: Optional[Any] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Stream agent response using official PydanticAI API.
        
        Args:
            agent: Database Agent record
            prompt: User prompt
            message_history: Previous conversation messages
            deps: Dependencies for agent execution
            **kwargs: Additional arguments for agent.run_stream()
            
        Returns:
            Agent stream result or None if failed
        """
        
        pydantic_agent = self.get_instantiated_agent(agent)
        if not pydantic_agent:
            return None
        
        try:
            # Use official PydanticAI agent.run_stream() API
            run_kwargs: Dict[str, Any] = {
                "user_prompt": prompt,
            }
            
            if message_history:
                run_kwargs["message_history"] = message_history
            
            if deps:
                run_kwargs["deps"] = deps
            
            # Add any additional kwargs
            run_kwargs.update(kwargs)
            
            # Import here to avoid dependency issues during development
            # PydanticAI run_stream returns a context manager, not a direct async generator
            stream_context = pydantic_agent.run_stream(**run_kwargs)
            
            # Update usage count
            agent.usage_count += 1
            
            return stream_context
            
        except Exception as e:
            print(f"Error streaming agent {agent.id}: {e}")
            return None
    
    def clear_agent_cache(self, agent_id: Optional[int] = None) -> None:
        """
        Clear agent cache for a specific agent or all agents.
        
        Args:
            agent_id: Specific agent ID to clear, or None to clear all
        """
        
        if agent_id:
            self._agent_cache.pop(agent_id, None)
        else:
            self._agent_cache.clear()
    
    async def get_agent_stats(self, db: AsyncSession, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Args:
            db: Database session
            user_id: User ID to filter by (None for global stats)
            
        Returns:
            Dictionary with agent statistics
        """
        
        query = select(Agent).where(Agent.is_active == True)
        
        if user_id:
            query = query.where(Agent.owner_id == user_id)
        
        result = await db.execute(query)
        agents = list(result.scalars().all())
        
        stats = {
            "total_agents": len(agents),
            "public_agents": sum(1 for a in agents if a.is_public),
            "private_agents": sum(1 for a in agents if not a.is_public),
            "total_usage": sum(a.usage_count for a in agents),
            "providers": {},
        }
        
        # Provider breakdown
        for agent in agents:
            provider = agent.model_provider.value if hasattr(agent.model_provider, 'value') else agent.model_provider
            if provider not in stats["providers"]:
                stats["providers"][provider] = 0
            stats["providers"][provider] += 1
        
        return stats


# Global agent service instance
agent_service = AgentService()