"""
Workflow API routes for graph-based agent orchestration.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...services import workflow_service
from ..schemas import (
    WorkflowCreate, WorkflowResponse, WorkflowUpdate,
    PaginationParams
)

router = APIRouter()


@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new workflow definition.
    
    Args:
        workflow_data: Workflow configuration
        db: Database session
        
    Returns:
        Created workflow
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    try:
        workflow = await workflow_service.create_workflow(
            name=workflow_data.name,
            description=workflow_data.description,
            workflow_type=workflow_data.workflow_type,
            graph_config=workflow_data.graph_config,
            initial_state=workflow_data.initial_state,
            user_id=user_id,
            db=db,
        )
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create workflow"
            )
        
        return WorkflowResponse.model_validate(workflow)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    List workflows with pagination.
    
    Args:
        pagination: Pagination parameters
        db: Database session
        
    Returns:
        List of workflows
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    workflows = await workflow_service.list_workflows(
        user_id=user_id,
        db=db,
        skip=pagination.skip,
        limit=pagination.limit,
    )
    
    return [
        WorkflowResponse.model_validate(workflow)
        for workflow in workflows
    ]


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get workflow by ID.
    
    Args:
        workflow_id: Workflow ID
        db: Database session
        
    Returns:
        Workflow details
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    workflow = await workflow_service.get_workflow(workflow_id, user_id, db)
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    return WorkflowResponse.model_validate(workflow)


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: int,
    workflow_data: WorkflowUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update workflow configuration.
    
    Args:
        workflow_id: Workflow ID
        workflow_data: Updated workflow data
        db: Database session
        
    Returns:
        Updated workflow
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    # Convert to dict and filter None values
    updates = {k: v for k, v in workflow_data.dict().items() if v is not None}
    
    workflow = await workflow_service.update_workflow(
        workflow_id=workflow_id,
        user_id=user_id,
        db=db,
        **updates
    )
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or unauthorized"
        )
    
    return WorkflowResponse.model_validate(workflow)


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete workflow.
    
    Args:
        workflow_id: Workflow ID
        db: Database session
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    success = await workflow_service.delete_workflow(workflow_id, user_id, db)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found or unauthorized"
        )


@router.post("/{workflow_id}/execute", response_model=dict)
async def execute_workflow(
    workflow_id: int,
    input_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a workflow.
    
    Args:
        workflow_id: Workflow ID
        input_data: Input data for workflow execution
        db: Database session
        
    Returns:
        Execution results
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    try:
        result = await workflow_service.execute_workflow(
            workflow_id=workflow_id,
            input_data=input_data,
            user_id=user_id,
            db=db,
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.get("/{workflow_id}/history", response_model=List[dict])
async def get_workflow_execution_history(
    workflow_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get workflow execution history.
    
    Args:
        workflow_id: Workflow ID
        db: Database session
        
    Returns:
        Execution history
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    history = await workflow_service.get_workflow_execution_history(
        workflow_id=workflow_id,
        user_id=user_id,
        db=db,
    )
    
    return history


@router.get("/templates/simple", response_model=dict)
async def get_simple_workflow_template():
    """Get a simple workflow template"""
    
    return {
        "name": "Simple Sequential Workflow",
        "description": "A simple workflow that runs agents in sequence",
        "workflow_type": "sequential",
        "graph_config": workflow_service.create_simple_workflow_config(
            agent_ids=[1, 2],  # Default agent IDs
            workflow_type="sequential"
        ),
        "initial_state": {
            "input": "Your input here",
            "step": 0
        }
    }


@router.get("/templates/conditional", response_model=dict)
async def get_conditional_workflow_template():
    """Get a conditional workflow template"""
    
    return {
        "name": "Conditional Workflow",
        "description": "A workflow with conditional branching",
        "workflow_type": "conditional",
        "graph_config": {
            "type": "conditional",
            "start_node": {
                "type": "agent",
                "agent_id": 1,
                "prompt_template": "Analyze this input: {input}"
            },
            "nodes": [
                {
                    "id": "decision",
                    "type": "conditional",
                    "condition_field": "agent_1_result",
                    "conditions": {
                        "positive": "positive_path",
                        "negative": "negative_path"
                    }
                }
            ]
        },
        "initial_state": {
            "input": "Your input here"
        }
    }