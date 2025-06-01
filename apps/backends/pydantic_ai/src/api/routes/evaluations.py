"""
Evaluation API routes for agent testing and benchmarking.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_db
from ...services import evaluation_service
from ..schemas import (
    EvaluationCreate, EvaluationResponse, BenchmarkRequest, 
    PaginationParams
)

router = APIRouter()


@router.post("/", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    evaluation_data: EvaluationCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create and run a new agent evaluation.
    
    Args:
        evaluation_data: Evaluation configuration
        db: Database session
        
    Returns:
        Evaluation results
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    try:
        # Provide default test cases if none provided
        test_cases = evaluation_data.test_cases or [
            {
                'inputs': {'prompt': 'Hello, how are you?'},
                'expected_output': 'A friendly greeting response',
                'metadata': {'category': 'greeting'}
            }
        ]
        
        # Create dataset from test cases
        dataset = evaluation_service.create_basic_evaluation_dataset(
            test_cases
        )
        
        # Run evaluation
        result = await evaluation_service.evaluate_agent(
            agent_id=evaluation_data.agent_id,
            evaluation_dataset=dataset,
            user_id=user_id,
            db=db,
            evaluation_name=evaluation_data.name,
            description=evaluation_data.description,
        )
        
        from datetime import datetime
        from ...database.models import EvaluationStatus
        
        return EvaluationResponse(
            id=result['evaluation_id'],
            name=evaluation_data.name,
            description=evaluation_data.description,
            status=EvaluationStatus.COMPLETED,
            agent_id=evaluation_data.agent_id,
            dataset_config=evaluation_data.dataset_config,
            evaluator_config=evaluation_data.evaluator_config,
            results=result['detailed_results'],
            score=result['evaluation_summary'].get('average_score', 0.0),
            total_cases=result['evaluation_summary'].get('total_cases', 0),
            passed_cases=result['evaluation_summary'].get('passed_cases', 0),
            failed_cases=result['evaluation_summary'].get('failed_cases', 0),
            started_at=None,  # Would be filled from database
            completed_at=None,  # Would be filled from database
            duration_seconds=result['evaluation_summary'].get('execution_time', 0.0),
            created_at=datetime.utcnow(),  # Use current time as placeholder
            created_by=user_id,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/", response_model=List[EvaluationResponse])
async def list_evaluations(
    pagination: PaginationParams = Depends(),
    agent_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List evaluations with pagination.
    
    Args:
        pagination: Pagination parameters
        agent_id: Filter by agent ID
        db: Database session
        
    Returns:
        List of evaluations
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    evaluations = await evaluation_service.list_evaluations(
        user_id=user_id,
        db=db,
        agent_id=agent_id,
        skip=pagination.skip,
        limit=pagination.limit,
    )
    
    return [
        EvaluationResponse(
            id=eval['id'],
            name=eval['name'],
            description=eval['description'],
            status=eval['status'],
            agent_id=eval['agent_id'],
            dataset_config={},  # Would be filled from database
            evaluator_config={},  # Would be filled from database
            results=eval.get('results', {}),
            score=eval['score'],
            total_cases=eval['total_cases'],
            passed_cases=eval['passed_cases'],
            failed_cases=eval['failed_cases'],
            started_at=eval['started_at'],
            completed_at=eval['completed_at'],
            duration_seconds=eval['duration_seconds'],
            created_at=eval['created_at'],
            created_by=user_id,
        )
        for eval in evaluations
    ]


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation by ID.
    
    Args:
        evaluation_id: Evaluation ID
        db: Database session
        
    Returns:
        Evaluation details
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    evaluation = await evaluation_service.get_evaluation_results(
        evaluation_id=evaluation_id,
        user_id=user_id,
        db=db,
    )
    
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation not found"
        )
    
    return EvaluationResponse(
        id=evaluation['id'],
        name=evaluation['name'],
        description=evaluation['description'],
        status=evaluation['status'],
        agent_id=evaluation['agent_id'],
        dataset_config={},  # Would be filled from database
        evaluator_config={},  # Would be filled from database
        results=evaluation.get('results', {}),
        score=evaluation['score'],
        total_cases=evaluation['total_cases'],
        passed_cases=evaluation['passed_cases'],
        failed_cases=evaluation['failed_cases'],
        started_at=evaluation['started_at'],
        completed_at=evaluation['completed_at'],
        duration_seconds=evaluation['duration_seconds'],
        created_at=evaluation['created_at'],
        created_by=user_id,
    )


@router.post("/benchmark", response_model=dict)
async def create_benchmark(
    benchmark_request: BenchmarkRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a benchmark comparison between multiple agents.
    
    Args:
        benchmark_request: Benchmark configuration
        db: Database session
        
    Returns:
        Benchmark results
    """
    
    # For demo, use user_id=1 (admin user)
    user_id = 1
    
    try:
        result = await evaluation_service.create_agent_benchmark(
            agent_ids=benchmark_request.agent_ids,
            test_cases=benchmark_request.test_cases,
            user_id=user_id,
            db=db,
            benchmark_name=getattr(benchmark_request, 'name', 'Agent Benchmark'),
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )


@router.get("/datasets/examples", response_model=dict)
async def get_example_datasets():
    """Get example evaluation datasets"""
    
    return {
        "basic_qa": {
            "description": "Basic question-answering evaluation",
            "test_cases": [
                {
                    "inputs": {"question": "What is the capital of France?"},
                    "expected_output": "Paris",
                    "metadata": {"difficulty": "easy", "category": "geography"}
                },
                {
                    "inputs": {"question": "What is 2 + 2?"},
                    "expected_output": "4",
                    "metadata": {"difficulty": "easy", "category": "math"}
                }
            ]
        },
        "tool_usage": {
            "description": "Tool usage evaluation",
            "test_cases": [
                {
                    "inputs": {"task": "Search for recent news about AI"},
                    "expected_output": {"expected_tools": ["web_search"], "contains": "AI"},
                    "metadata": {"requires_tools": True, "category": "search"}
                }
            ]
        }
    }