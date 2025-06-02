# 🔧 Agent System & Dynamic Creation

> **Dynamic agent creation and tool system using official Pydantic AI patterns**

## 🎯 Overview

The Agent System provides dynamic agent creation, tool registration, and agent lifecycle management using official Pydantic AI API patterns. Agents can be created at runtime with custom configurations and tools.

## 🤖 Dynamic Agent Factory

### Core Agent Factory Implementation
```python
# src/agents/dynamic/factory.py
from pydantic_ai import Agent, RunContext
from typing import Dict, Any, List, Optional, Callable
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ...tools.filesystem.read_write import FileSystemTools
from ...services.model_service import ModelService
from ...models.agent import Agent as AgentModel

class AgentDefinition(BaseModel):
    """Agent definition schema for dynamic creation"""
    
    name: str
    display_name: str
    description: str
    model: Dict[str, Any]  # Model configuration
    system_prompt: str
    tools: List[Dict[str, Any]]  # Tool configurations
    metadata: Dict[str, Any] = {}

class PydanticAgentFactory:
    """Factory for creating Pydantic AI agents using official API"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.filesystem_tools = FileSystemTools()
        self._tool_registry: Dict[str, Callable] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
    
    async def create_agent(
        self, 
        definition: AgentDefinition,
        deps_type: Optional[type] = None
    ) -> Agent:
        """Create a Pydantic AI agent using official API patterns"""
        
        # Get model using official API
        model = self.model_service.get_model(
            provider=definition.model["provider"],
            model_name=definition.model["model"],
            temperature=definition.model.get("temperature", 0.1),
            max_tokens=definition.model.get("max_tokens"),
        )
        
        # Create agent with official Pydantic AI constructor
        agent = Agent(
            model=model,
            system_prompt=definition.system_prompt,
            deps_type=deps_type or type(None)
        )
        
        # Register tools using official decorators
        self._register_tools(agent, definition.tools)
        
        return agent
    
    def _register_tools(self, agent: Agent, tool_configs: List[Dict[str, Any]]):
        """Register tools with agent using official patterns"""
        
        for tool_config in tool_configs:
            if not tool_config.get("enabled", True):
                continue
                
            tool_name = tool_config["name"]
            
            if tool_name in self._tool_registry:
                # Register the tool function with the agent
                tool_func = self._tool_registry[tool_name]
                
                # Use official @agent.tool decorator pattern
                if tool_config.get("plain", False):
                    # Use @agent.tool_plain for simple tools
                    decorated_tool = agent.tool_plain(tool_func)
                else:
                    # Use @agent.tool for context-aware tools
                    decorated_tool = agent.tool(tool_func)
                
                # Store reference to decorated tool
                setattr(agent, f"_{tool_name}_tool", decorated_tool)
    
    def _register_builtin_tools(self):
        """Register built-in tools in the tool registry"""
        
        # File system tools
        self._tool_registry["file_read"] = self._create_file_read_tool()
        self._tool_registry["file_write"] = self._create_file_write_tool()
        self._tool_registry["grep_search"] = self._create_grep_tool()
        
        # Add more built-in tools here
        self._tool_registry["web_search"] = self._create_web_search_tool()
        self._tool_registry["calculator"] = self._create_calculator_tool()
    
    def _create_file_read_tool(self):
        """Create file reading tool using official patterns"""
        
        async def read_file(ctx: RunContext[None], file_path: str) -> str:
            """Read contents of a file"""
            try:
                return await self.filesystem_tools.read_file(file_path)
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        return read_file
    
    def _create_file_write_tool(self):
        """Create file writing tool using official patterns"""
        
        async def write_file(
            ctx: RunContext[None], 
            file_path: str, 
            content: str
        ) -> str:
            """Write content to a file"""
            try:
                await self.filesystem_tools.write_file(file_path, content)
                return f"Successfully wrote to {file_path}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        return write_file
    
    def _create_grep_tool(self):
        """Create grep search tool using official patterns"""
        
        async def grep_search(
            ctx: RunContext[None], 
            pattern: str, 
            directory: str = "."
        ) -> str:
            """Search for patterns in files"""
            try:
                results = await self.filesystem_tools.grep_search(pattern, directory)
                return f"Found {len(results)} matches:\n" + "\n".join(results)
            except Exception as e:
                return f"Error searching: {str(e)}"
        
        return grep_search
    
    def _create_web_search_tool(self):
        """Create web search tool (example)"""
        
        async def web_search(ctx: RunContext[None], query: str) -> str:
            """Search the web for information"""
            # This would integrate with a web search API
            return f"Web search results for: {query}"
        
        return web_search
    
    def _create_calculator_tool(self):
        """Create calculator tool (example)"""
        
        def calculate(ctx: RunContext[None], expression: str) -> str:
            """Perform mathematical calculations"""
            try:
                # Safe evaluation of mathematical expressions
                import ast
                import operator
                
                # Define allowed operations
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }
                
                def eval_expr(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](eval_expr(node.operand))
                    else:
                        raise ValueError(f"Unsupported operation: {type(node)}")
                
                result = eval_expr(ast.parse(expression, mode='eval').body)
                return f"Result: {result}"
                
            except Exception as e:
                return f"Error calculating: {str(e)}"
        
        return calculate
    
    async def store_agent_definition(
        self,
        definition: AgentDefinition,
        user_id: UUID,
        db: AsyncSession
    ) -> AgentModel:
        """Store agent definition in database"""
        
        agent_model = AgentModel(
            name=definition.name,
            display_name=definition.display_name,
            description=definition.description,
            owner_id=user_id,
            definition=definition.dict(),
            metadata=definition.metadata
        )
        
        db.add(agent_model)
        await db.commit()
        await db.refresh(agent_model)
        
        return agent_model
    
    def register_custom_tool(self, name: str, tool_func: Callable):
        """Register a custom tool in the factory"""
        self._tool_registry[name] = tool_func
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self._tool_registry.keys())
```

## 🛠️ File System Tools Implementation

### Core File System Tools
```python
# src/tools/filesystem/read_write.py
import os
import asyncio
import aiofiles
from typing import List, Optional
from pathlib import Path

class FileSystemTools:
    """File system operations for agents"""
    
    def __init__(self, base_path: str = "/tmp/agent_workspace"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def read_file(self, file_path: str) -> str:
        """Read file contents safely"""
        
        # Ensure path is within base directory for security
        full_path = self._secure_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    async def write_file(self, file_path: str, content: str) -> None:
        """Write content to file safely"""
        
        full_path = self._secure_path(file_path)
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def list_directory(self, directory_path: str = ".") -> List[str]:
        """List directory contents"""
        
        full_path = self._secure_path(directory_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not full_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        items = []
        for item in full_path.iterdir():
            item_type = "dir" if item.is_dir() else "file"
            items.append(f"{item_type}: {item.name}")
        
        return sorted(items)
    
    async def grep_search(
        self, 
        pattern: str, 
        directory: str = ".",
        file_extension: Optional[str] = None
    ) -> List[str]:
        """Search for patterns in files"""
        
        import re
        
        full_path = self._secure_path(directory)
        
        if not full_path.exists() or not full_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        results = []
        pattern_regex = re.compile(pattern, re.IGNORECASE)
        
        for file_path in full_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Filter by file extension if specified
            if file_extension and not file_path.suffix == file_extension:
                continue
            
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                for line_num, line in enumerate(content.splitlines(), 1):
                    if pattern_regex.search(line):
                        relative_path = file_path.relative_to(self.base_path)
                        results.append(f"{relative_path}:{line_num}: {line.strip()}")
                        
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we can't read
                continue
        
        return results
    
    def _secure_path(self, path: str) -> Path:
        """Ensure path is within base directory (security)"""
        
        full_path = (self.base_path / path).resolve()
        
        # Ensure the resolved path is still within base_path
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(f"Path outside allowed directory: {path}")
        
        return full_path
    
    async def create_directory(self, directory_path: str) -> None:
        """Create directory safely"""
        
        full_path = self._secure_path(directory_path)
        full_path.mkdir(parents=True, exist_ok=True)
    
    async def delete_file(self, file_path: str) -> None:
        """Delete file safely"""
        
        full_path = self._secure_path(file_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if full_path.is_file():
            full_path.unlink()
        else:
            raise ValueError(f"Path is not a file: {file_path}")
```

## 🔄 Agent Service Integration

### Agent Service with Dynamic Creation
```python
# src/services/agent_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional
from uuid import UUID
from pydantic_ai import Agent

from ..models.agent import Agent as AgentModel
from ..schemas.agents import AgentCreate, AgentResponse
from ..agents.dynamic.factory import PydanticAgentFactory, AgentDefinition
from .model_service import ModelService

class AgentService:
    """Service for managing agents using official Pydantic AI API"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.agent_factory = PydanticAgentFactory(model_service)
        self._agent_cache: Dict[UUID, Agent] = {}
    
    async def create_agent(
        self,
        agent_data: AgentCreate,
        user_id: UUID,
        db: AsyncSession
    ) -> AgentResponse:
        """Create a new dynamic agent"""
        
        # Convert to internal definition format
        definition = AgentDefinition(
            name=agent_data.name,
            display_name=agent_data.display_name,
            description=agent_data.description,
            model=agent_data.model.dict(),
            system_prompt=agent_data.system_prompt,
            tools=[tool.dict() for tool in agent_data.tools],
            metadata=agent_data.metadata
        )
        
        # Validate agent definition
        await self._validate_agent_definition(definition)
        
        # Store in database
        agent_model = await self.agent_factory.store_agent_definition(
            definition, user_id, db
        )
        
        return AgentResponse.from_orm(agent_model)
    
    async def get_agent_instance(self, agent_id: UUID, db: AsyncSession) -> Agent:
        """Get agent instance with caching for conversation continuity"""
        
        if agent_id not in self._agent_cache:
            agent_model = await self._get_agent_model(agent_id, db)
            definition = AgentDefinition(**agent_model.definition)
            pydantic_agent = await self.agent_factory.create_agent(definition)
            self._agent_cache[agent_id] = pydantic_agent
        
        return self._agent_cache[agent_id]
    
    async def update_agent(
        self,
        agent_id: UUID,
        updates: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> AgentResponse:
        """Update existing agent configuration"""
        
        agent_model = await self._get_agent_model(agent_id, db)
        
        # Verify ownership
        if agent_model.owner_id != user_id:
            raise ValueError("Not authorized to update this agent")
        
        # Update fields
        for field, value in updates.items():
            if hasattr(agent_model, field):
                setattr(agent_model, field, value)
        
        # Clear cache to force reload
        if agent_id in self._agent_cache:
            del self._agent_cache[agent_id]
        
        await db.commit()
        await db.refresh(agent_model)
        
        return AgentResponse.from_orm(agent_model)
    
    async def delete_agent(
        self,
        agent_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> bool:
        """Delete agent and clear cache"""
        
        agent_model = await self._get_agent_model(agent_id, db)
        
        # Verify ownership
        if agent_model.owner_id != user_id:
            raise ValueError("Not authorized to delete this agent")
        
        # Clear cache
        if agent_id in self._agent_cache:
            del self._agent_cache[agent_id]
        
        await db.delete(agent_model)
        await db.commit()
        
        return True
    
    async def list_user_agents(
        self,
        user_id: UUID,
        db: AsyncSession,
        include_public: bool = False
    ) -> List[AgentResponse]:
        """List agents owned by user"""
        
        from sqlalchemy import select, or_
        
        query = select(AgentModel).where(AgentModel.owner_id == user_id)
        
        if include_public:
            query = query.where(
                or_(AgentModel.owner_id == user_id, AgentModel.is_public == True)
            )
        
        result = await db.execute(query)
        agents = result.scalars().all()
        
        return [AgentResponse.from_orm(agent) for agent in agents]
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return self.agent_factory.get_available_tools()
    
    async def _validate_agent_definition(self, definition: AgentDefinition):
        """Validate agent definition before creation"""
        
        # Validate model configuration
        model_config = definition.model
        if not self.model_service.validate_model_config(
            model_config["provider"], 
            model_config["model"]
        ):
            raise ValueError(f"Invalid model configuration: {model_config}")
        
        # Validate tools
        available_tools = self.agent_factory.get_available_tools()
        for tool_config in definition.tools:
            tool_name = tool_config["name"]
            if tool_name not in available_tools:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        return True
    
    async def _get_agent_model(self, agent_id: UUID, db: AsyncSession) -> AgentModel:
        """Get agent model from database"""
        from sqlalchemy import select
        
        result = await db.execute(select(AgentModel).where(AgentModel.id == agent_id))
        agent_model = result.scalar_one_or_none()
        
        if not agent_model:
            raise ValueError(f"Agent not found: {agent_id}")
        
        return agent_model
```

## 🎯 Agent Configuration Schema

### Agent Creation Schemas
```python
# src/schemas/agents.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from uuid import UUID
from datetime import datetime

class ToolConfig(BaseModel):
    """Tool configuration schema"""
    
    name: str
    enabled: bool = True
    plain: bool = False  # Use @agent.tool_plain vs @agent.tool
    config: Dict[str, Any] = {}

class ModelConfig(BaseModel):
    """Model configuration schema"""
    
    provider: Literal["google", "anthropic"]
    model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)

class AgentCreate(BaseModel):
    """Schema for creating new agents"""
    
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    system_prompt: str = Field(..., min_length=1)
    model: ModelConfig
    tools: List[ToolConfig] = []
    metadata: Dict[str, Any] = {}
    is_public: bool = False

class AgentUpdate(BaseModel):
    """Schema for updating agents"""
    
    display_name: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    system_prompt: Optional[str] = None
    model: Optional[ModelConfig] = None
    tools: Optional[List[ToolConfig]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None

class AgentResponse(BaseModel):
    """Schema for agent responses"""
    
    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    owner_id: UUID
    definition: Dict[str, Any]
    metadata: Dict[str, Any]
    is_public: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ToolInfo(BaseModel):
    """Tool information schema"""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    examples: List[str] = []
```

## 🔧 Advanced Agent Features

### Agent Templates System
```python
# src/agents/templates/manager.py
from typing import Dict, List
from ..dynamic.factory import AgentDefinition

class AgentTemplateManager:
    """Manage predefined agent templates"""
    
    def __init__(self):
        self._templates: Dict[str, AgentDefinition] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in agent templates"""
        
        # File Assistant Template
        self._templates["file_assistant"] = AgentDefinition(
            name="file_assistant",
            display_name="File Assistant",
            description="Helps with file operations and code management",
            model={
                "provider": "google",
                "model": "gemini-1.5-flash",
                "temperature": 0.1
            },
            system_prompt=(
                "You are a helpful file assistant. You can read, write, and search "
                "through files to help users manage their projects. Always be careful "
                "with file operations and confirm before making destructive changes."
            ),
            tools=[
                {"name": "file_read", "enabled": True},
                {"name": "file_write", "enabled": True},
                {"name": "grep_search", "enabled": True},
                {"name": "list_directory", "enabled": True}
            ],
            metadata={"category": "productivity", "template": True}
        )
        
        # Code Review Template
        self._templates["code_reviewer"] = AgentDefinition(
            name="code_reviewer",
            display_name="Code Reviewer",
            description="Reviews code for best practices and potential issues",
            model={
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.0
            },
            system_prompt=(
                "You are an expert code reviewer. Analyze code for:\n"
                "- Best practices and code quality\n"
                "- Security vulnerabilities\n"
                "- Performance issues\n"
                "- Maintainability concerns\n"
                "Provide constructive feedback with specific suggestions."
            ),
            tools=[
                {"name": "file_read", "enabled": True},
                {"name": "grep_search", "enabled": True}
            ],
            metadata={"category": "development", "template": True}
        )
        
        # Research Assistant Template
        self._templates["research_assistant"] = AgentDefinition(
            name="research_assistant",
            display_name="Research Assistant",
            description="Helps with research and information gathering",
            model={
                "provider": "google",
                "model": "gemini-1.5-pro",
                "temperature": 0.3
            },
            system_prompt=(
                "You are a research assistant. Help users gather, analyze, and "
                "synthesize information. Always cite sources when possible and "
                "provide well-structured, comprehensive responses."
            ),
            tools=[
                {"name": "web_search", "enabled": True},
                {"name": "file_read", "enabled": True},
                {"name": "file_write", "enabled": True}
            ],
            metadata={"category": "research", "template": True}
        )
    
    def get_template(self, template_name: str) -> Optional[AgentDefinition]:
        """Get agent template by name"""
        return self._templates.get(template_name)
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List available templates"""
        return [
            {
                "name": name,
                "display_name": template.display_name,
                "description": template.description,
                "category": template.metadata.get("category", "general")
            }
            for name, template in self._templates.items()
        ]
    
    def create_from_template(
        self, 
        template_name: str, 
        customizations: Optional[Dict[str, Any]] = None
    ) -> AgentDefinition:
        """Create agent definition from template with customizations"""
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Create copy of template
        agent_def = AgentDefinition(**template.dict())
        
        # Apply customizations
        if customizations:
            for key, value in customizations.items():
                if hasattr(agent_def, key):
                    setattr(agent_def, key, value)
        
        return agent_def

template_manager = AgentTemplateManager()
```

## 🚀 Key Features

**🤖 Dynamic Agent Creation:**
- [`Agent(model, system_prompt)`](https://docs.pydantic.ai/agents/#pydantic_ai.Agent) constructor using official API
- Runtime agent configuration from database
- Model provider flexibility (Google Vertex AI, Anthropic)

**🛠️ Tool System:**
- [`@agent.tool`](https://docs.pydantic.ai/tools/#function-tools) and [`@agent.tool_plain`](https://docs.pydantic.ai/tools/#function-tools) decorators
- [`RunContext`](https://docs.pydantic.ai/tools/#run-context) for context-aware tools
- Secure file system operations with sandboxing

**🔄 Agent Lifecycle Management:**
- Database persistence of agent definitions
- Instance caching for performance
- Template system for common agent types

**⚡ Performance & Security:**
- Agent instance caching for conversation continuity
- Secure file operations within sandboxed directories
- Validation of tool and model configurations

**🔧 Extensibility:**
- Plugin system for custom tools
- Template-based agent creation
- Configuration validation and testing

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*