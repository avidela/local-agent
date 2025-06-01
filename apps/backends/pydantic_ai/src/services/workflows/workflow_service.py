"""
Workflow service for graph-based agent orchestration using pydantic_graph.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, cast
from uuid import uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.in_mem import SimpleStatePersistence

from ...database import Workflow, WorkflowExecution, Agent, User, WorkflowStatus
from ...config import settings
from ..agents import agent_service
from ..models import model_provider_service
from ...observability import trace_workflow_execution


@dataclass
class WorkflowState:
    """Base state for workflow execution"""
    workflow_id: int
    user_id: int
    session_data: Dict[str, Any]
    current_step: int = 0
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.execution_history is None:
            self.execution_history = []


class AgentNode(BaseNode):
    """Node that executes a PydanticAI agent"""
    
    def __init__(self, agent_id: int, prompt_template: str, next_nodes: Optional[Dict[str, str]] = None):
        self.agent_id = agent_id
        self.prompt_template = prompt_template
        self.next_nodes = next_nodes if next_nodes is not None else {}
    
    async def run(self, ctx: GraphRunContext[WorkflowState, None]) -> Union['AgentNode', 'End', Any]:
        """Execute the agent and determine next node"""
        
        with trace_workflow_execution(f"workflow_{ctx.state.workflow_id}", f"agent_{self.agent_id}"):
            try:
                # Get agent from database
                from ...database import get_db_session
                async with get_db_session() as db:
                    agent = await agent_service.get_agent(db, self.agent_id, ctx.state.user_id)
                    if not agent:
                        raise ValueError(f"Agent {self.agent_id} not found")
                    
                    # Format prompt with state data
                    prompt = self.prompt_template.format(**ctx.state.session_data)
                    
                    # Execute agent
                    result = await agent_service.run_agent(agent, prompt)
                    
                    # Store execution result
                    execution_record = {
                        'step': ctx.state.current_step,
                        'node_type': 'agent',
                        'agent_id': self.agent_id,
                        'prompt': prompt,
                        'result': str(result) if result else None,
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    ctx.state.execution_history.append(execution_record)
                    ctx.state.current_step += 1
                    
                    # Store result in session data
                    ctx.state.session_data[f'agent_{self.agent_id}_result'] = str(result) if result else None
                    
                    # Determine next node based on result or configuration
                    if result and self.next_nodes:
                        # Simple routing based on result content
                        result_str = str(result).lower()
                        for condition, next_node in self.next_nodes.items():
                            if condition.lower() in result_str:
                                # Would need to instantiate the actual next node
                                # For now, return End
                                return End(result)
                    
                    return End(result)
                    
            except Exception as e:
                # Record error
                error_record = {
                    'step': ctx.state.current_step,
                    'node_type': 'agent_error',
                    'agent_id': self.agent_id,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                }
                ctx.state.execution_history.append(error_record)
                raise


class ConditionalNode(BaseNode):
    """Node that routes based on conditions"""
    
    def __init__(self, condition_field: str, conditions: Dict[str, str]):
        self.condition_field = condition_field
        self.conditions = conditions
    
    async def run(self, ctx: GraphRunContext[WorkflowState, None]) -> Union['AgentNode', 'ConditionalNode', 'End']:
        """Evaluate condition and route to next node"""
        
        value = ctx.state.session_data.get(self.condition_field, '')
        value_str = str(value).lower()
        
        # Record decision
        decision_record = {
            'step': ctx.state.current_step,
            'node_type': 'conditional',
            'condition_field': self.condition_field,
            'value': value_str,
            'timestamp': datetime.utcnow().isoformat(),
        }
        ctx.state.execution_history.append(decision_record)
        ctx.state.current_step += 1
        
        # Find matching condition
        for condition, next_node in self.conditions.items():
            if condition.lower() in value_str:
                # Would need to instantiate the actual next node
                # For now, return End with the routing decision
                return End(None)
        
        # Default case
        return End(None)


class WorkflowService:
    """Service for managing graph-based workflows with pydantic_graph"""

    def __init__(self):
        """Initialize workflow service"""
        self.agent_service = agent_service
        self.model_service = model_provider_service

    async def create_workflow(
        self,
        name: str,
        description: Optional[str],
        workflow_type: str,
        graph_config: Dict[str, Any],
        initial_state: Dict[str, Any],
        user_id: int,
        db: AsyncSession,
    ) -> Optional[Workflow]:
        """
        Create a new workflow definition.
        
        Args:
            name: Workflow name
            description: Workflow description
            workflow_type: Type of workflow (graph, sequential, etc.)
            graph_config: Graph configuration with nodes and edges
            initial_state: Initial state data
            user_id: User creating the workflow
            db: Database session
            
        Returns:
            Created Workflow instance
        """
        
        workflow = Workflow(
            name=name,
            description=description,
            workflow_type=workflow_type,
            graph_config=graph_config,
            initial_state=initial_state,
            current_state=initial_state.copy(),
            created_by=user_id,
        )
        
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
        
        return workflow

    async def execute_workflow(
        self,
        workflow_id: int,
        input_data: Dict[str, Any],
        user_id: int,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Execute a workflow from start to finish.
        
        Args:
            workflow_id: Workflow ID to execute
            input_data: Input data for the workflow
            user_id: User executing the workflow
            db: Database session
            
        Returns:
            Execution results
        """
        
        # Get workflow
        workflow = await self.get_workflow(workflow_id, user_id, db)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Update workflow status
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        await db.commit()
        
        try:
            # Create workflow state
            state = WorkflowState(
                workflow_id=workflow_id,
                user_id=user_id,
                session_data=input_data,
            )
            
            # Build graph from configuration
            graph = self._build_graph_from_config(workflow.graph_config)
            start_node = self._create_start_node_from_config(workflow.graph_config)
            
            # Execute workflow
            with trace_workflow_execution(workflow.name, "full_execution"):
                result = await graph.run(start_node)
                
                # Update workflow with results
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.utcnow()
                workflow.current_state = {
                    'session_data': state.session_data,
                    'execution_history': state.execution_history,
                    'final_result': str(result.output) if result else None,
                }
                
                # Store execution steps
                for i, step in enumerate(state.execution_history):
                    execution = WorkflowExecution(
                        workflow_id=workflow_id,
                        step_number=i,
                        node_name=step.get('node_type', 'unknown'),
                        node_config=step,
                        execution_result={'result': step.get('result', step.get('error'))},
                        started_at=datetime.fromisoformat(step['timestamp']),
                        completed_at=datetime.fromisoformat(step['timestamp']),
                    )
                    db.add(execution)
                
                await db.commit()
                
                return {
                    'workflow_id': workflow_id,
                    'status': 'completed',
                    'result': result.output if result else None,
                    'execution_history': state.execution_history,
                    'duration_seconds': (
                        cast(datetime, workflow.completed_at) - cast(datetime, workflow.started_at)
                    ).total_seconds() if workflow.completed_at and workflow.started_at else 0,
                }
                
        except Exception as e:
            # Update workflow with error
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            workflow.error_message = str(e)
            workflow.current_state = {
                'error': str(e),
                'session_data': input_data,
            }
            await db.commit()
            
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'error': str(e),
                'duration_seconds': (
                    cast(datetime, workflow.completed_at) - cast(datetime, workflow.started_at)
                ).total_seconds() if workflow.completed_at and workflow.started_at else 0,
            }

    async def get_workflow(
        self,
        workflow_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> Optional[Workflow]:
        """Get workflow by ID with user access validation"""
        
        query = select(Workflow).where(
            and_(
                Workflow.id == workflow_id,
                Workflow.created_by == user_id
            )
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_workflows(
        self,
        user_id: int,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Workflow]:
        """List workflows for a user"""
        
        query = select(Workflow).where(Workflow.created_by == user_id)
        query = query.offset(skip).limit(limit).order_by(Workflow.created_at.desc())
        
        result = await db.execute(query)
        return list(result.scalars().all())

    async def update_workflow(
        self,
        workflow_id: int,
        user_id: int,
        db: AsyncSession,
        **updates
    ) -> Optional[Workflow]:
        """Update workflow configuration"""
        
        workflow = await self.get_workflow(workflow_id, user_id, db)
        if not workflow:
            return None
        
        # Update allowed fields
        allowed_fields = {
            'name', 'description', 'workflow_type', 'graph_config', 
            'initial_state', 'current_state'
        }
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(workflow, field):
                setattr(workflow, field, value)
        
        await db.commit()
        await db.refresh(workflow)
        
        return workflow

    async def delete_workflow(
        self,
        workflow_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> bool:
        """Delete workflow (soft delete)"""
        
        workflow = await self.get_workflow(workflow_id, user_id, db)
        if not workflow:
            return False
        
        # For now, mark as paused (soft delete)
        workflow.status = WorkflowStatus.PAUSED
        await db.commit()
        
        return True

    async def get_workflow_execution_history(
        self,
        workflow_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Get execution history for a workflow"""
        
        # Verify access
        workflow = await self.get_workflow(workflow_id, user_id, db)
        if not workflow:
            return []
        
        query = select(WorkflowExecution).where(
            WorkflowExecution.workflow_id == workflow_id
        ).order_by(WorkflowExecution.step_number)
        
        result = await db.execute(query)
        executions = list(result.scalars().all())
        
        return [
            {
                'step_number': exec.step_number,
                'node_name': exec.node_name,
                'node_data': exec.node_data,
                'result_data': exec.result_data,
                'error_message': exec.error_message,
                'execution_time_ms': exec.execution_time_ms,
                'state_before': exec.state_before,
                'state_after': exec.state_after,
            }
            for exec in executions
        ]

    def _build_graph_from_config(self, graph_config: Dict[str, Any]) -> Graph:
        """Build pydantic_graph Graph from configuration"""
        
        # For now, create a simple graph with agent nodes
        # In a full implementation, this would parse the graph_config
        # and create the appropriate node types
        
        nodes = [AgentNode, ConditionalNode]  # Add other node types as needed
        
        return Graph(nodes=nodes)

    def _create_start_node_from_config(self, graph_config: Dict[str, Any]) -> BaseNode:
        """Create the starting node from configuration"""
        
        # Parse the graph_config to determine the start node
        start_config = graph_config.get('start_node', {})
        node_type = start_config.get('type', 'agent')
        
        if node_type == 'agent':
            agent_id = start_config.get('agent_id', 1)  # Default to first agent
            prompt_template = start_config.get('prompt_template', 'Execute task: {input}')
            return AgentNode(agent_id, prompt_template)
        
        # Default to a simple agent node
        return AgentNode(1, "Execute workflow: {input}")

    def create_simple_workflow_config(
        self,
        agent_ids: List[int],
        workflow_type: str = "sequential"
    ) -> Dict[str, Any]:
        """Create a simple workflow configuration"""
        
        if workflow_type == "sequential":
            # Create a sequential workflow with multiple agents
            return {
                'type': 'sequential',
                'start_node': {
                    'type': 'agent',
                    'agent_id': agent_ids[0] if agent_ids else 1,
                    'prompt_template': 'Start workflow: {input}'
                },
                'nodes': [
                    {
                        'id': f'agent_{agent_id}',
                        'type': 'agent',
                        'agent_id': agent_id,
                        'prompt_template': f'Process step with agent {agent_id}: {{previous_result}}'
                    }
                    for agent_id in agent_ids
                ],
                'edges': [
                    {'from': f'agent_{agent_ids[i]}', 'to': f'agent_{agent_ids[i+1]}'}
                    for i in range(len(agent_ids) - 1)
                ] if len(agent_ids) > 1 else []
            }
        
        # Default single agent workflow
        return {
            'type': 'single',
            'start_node': {
                'type': 'agent',
                'agent_id': agent_ids[0] if agent_ids else 1,
                'prompt_template': 'Execute: {input}'
            },
            'nodes': [],
            'edges': []
        }


# Global workflow service instance
workflow_service = WorkflowService()