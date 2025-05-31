# 🧪 Pydantic Evals - Model Evaluation & Testing

> **Systematic testing and evaluation framework for AI models and agents using official Pydantic Evals**

## 🎯 Overview

Pydantic Evals is a powerful evaluation framework designed to help you systematically test and evaluate the performance and accuracy of AI systems, especially when working with LLMs. It provides structured evaluation with datasets, cases, and evaluators.

## 📦 Installation

```bash
# Basic Pydantic Evals
uv add pydantic-evals

# With Logfire integration
uv add 'pydantic-evals[logfire]'

# Or with PydanticAI
uv add 'pydantic-ai-slim[evals]'
```

## 🗂️ Core Concepts

### Cases and Datasets

```python
# src/evaluation/dataset_manager.py
from pydantic_evals import Case, Dataset

class EvaluationDatasetManager:
    """Manager for creating and organizing evaluation datasets"""
    
    def create_basic_dataset(self) -> Dataset:
        """Create a basic evaluation dataset"""
        
        cases = [
            Case(
                name='simple_qa',
                inputs='What is the capital of France?',
                expected_output='Paris',
                metadata={'difficulty': 'easy', 'category': 'geography'}
            ),
            Case(
                name='complex_reasoning',
                inputs='If a train travels 120 km in 1.5 hours, what is its average speed?',
                expected_output='80 km/h',
                metadata={'difficulty': 'medium', 'category': 'math'}
            )
        ]
        
        return Dataset(cases=cases)
    
    def create_agent_dataset(self) -> Dataset:
        """Create dataset for agent evaluation"""
        
        from ...schemas.agents import AgentCreate, ModelConfig, ToolConfig
        
        cases = [
            Case(
                name='file_operations_agent',
                inputs={
                    "agent_config": AgentCreate(
                        name="file_agent",
                        display_name="File Operations Agent",
                        description="Agent for file operations",
                        model=ModelConfig(
                            provider="vertex-gemini",
                            model="gemini-1.5-flash",
                            temperature=0.1
                        ),
                        system_prompt="You are a helpful assistant for file operations.",
                        tools=[
                            ToolConfig(name="file_read", enabled=True),
                            ToolConfig(name="file_write", enabled=True)
                        ]
                    ),
                    "task": "Read the file 'example.txt' and summarize its contents"
                },
                expected_output="Successfully read and summarized file contents",
                metadata={'category': 'agent_tools', 'complexity': 'medium'}
            )
        ]
        
        return Dataset(cases=cases)
```

### Built-in Evaluators

```python
# src/evaluation/evaluators.py
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge
from typing import Any

class AgentResponseEvaluator(Evaluator[dict, str]):
    """Custom evaluator for agent responses"""
    
    def evaluate(self, ctx: EvaluatorContext[dict, str]) -> float:
        """Evaluate agent response quality"""
        
        if not isinstance(ctx.output, str):
            return 0.0
        
        # Check if response is meaningful (not empty, not error)
        if not ctx.output.strip():
            return 0.0
        
        if "error" in ctx.output.lower():
            return 0.2
        
        # Check if response contains expected keywords
        if ctx.expected_output:
            expected_words = ctx.expected_output.lower().split()
            output_words = ctx.output.lower().split()
            
            matches = sum(1 for word in expected_words if word in output_words)
            return min(1.0, matches / len(expected_words))
        
        return 0.8  # Default score for valid response

class ToolUsageEvaluator(Evaluator[dict, dict]):
    """Evaluator that checks if correct tools were used"""
    
    def evaluate(self, ctx: EvaluatorContext[dict, dict]) -> dict[str, Any]:
        """Evaluate tool usage from span tree"""
        
        if ctx.span_tree is None:
            return {'tools_used': False, 'correct_tools': 0.0}
        
        # Look for tool execution spans
        tool_spans = ctx.span_tree.find(lambda node: 'tool' in node.name.lower())
        
        expected_tools = ctx.inputs.get('expected_tools', [])
        used_tools = [span.name for span in tool_spans]
        
        if not expected_tools:
            return {'tools_used': len(used_tools) > 0, 'correct_tools': 1.0}
        
        correct_count = sum(1 for tool in expected_tools if any(tool in used for used in used_tools))
        score = correct_count / len(expected_tools) if expected_tools else 1.0
        
        return {
            'tools_used': len(used_tools) > 0,
            'correct_tools': score,
            'used_tools': used_tools,
            'expected_tools': expected_tools
        }
```

## 🔧 Agent Evaluation Service

```python
# src/services/evaluation_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
from uuid import UUID
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import IsInstance, LLMJudge

from ..models.agent import Agent as AgentModel
from ..services.agent_service import AgentService
from ..evaluation.evaluators import AgentResponseEvaluator, ToolUsageEvaluator

class AgentEvaluationService:
    """Service for evaluating PydanticAI agents"""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
    
    async def evaluate_agent(
        self,
        agent_id: UUID,
        evaluation_dataset: Dataset,
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Evaluate an agent using a dataset"""
        
        # Get agent instance
        agent = await self.agent_service.get_agent_instance(agent_id, db)
        
        # Create evaluation function
        async def agent_task(inputs: Dict[str, Any]) -> str:
            """Task function for agent evaluation"""
            
            message = inputs.get('message', inputs.get('task', str(inputs)))
            
            try:
                result = await agent.run(message)
                return result.data
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Run evaluation
        report = evaluation_dataset.evaluate_sync(agent_task)
        
        return {
            'agent_id': str(agent_id),
            'evaluation_summary': self._extract_summary(report),
            'detailed_results': self._extract_details(report)
        }
    
    async def create_agent_benchmark(
        self,
        agent_ids: List[UUID],
        test_cases: List[Dict[str, Any]],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Create benchmark comparing multiple agents"""
        
        # Convert test cases to evaluation dataset
        cases = [
            Case(
                name=f"case_{i}",
                inputs=case['inputs'],
                expected_output=case.get('expected_output'),
                metadata=case.get('metadata', {})
            )
            for i, case in enumerate(test_cases)
        ]
        
        dataset = Dataset(
            cases=cases,
            evaluators=[
                IsInstance(type_name='str'),
                AgentResponseEvaluator(),
                LLMJudge(
                    rubric='Response should be helpful, accurate, and relevant to the input',
                    include_input=True
                )
            ]
        )
        
        # Evaluate each agent
        results = {}
        for agent_id in agent_ids:
            agent_result = await self.evaluate_agent(agent_id, dataset, user_id, db)
            results[str(agent_id)] = agent_result
        
        return {
            'benchmark_summary': self._create_benchmark_summary(results),
            'agent_results': results
        }
    
    def create_tool_evaluation_dataset(self, tool_scenarios: List[Dict[str, Any]]) -> Dataset:
        """Create dataset specifically for tool evaluation"""
        
        cases = []
        for scenario in tool_scenarios:
            case = Case(
                name=scenario['name'],
                inputs={
                    'task': scenario['task'],
                    'expected_tools': scenario.get('expected_tools', [])
                },
                expected_output=scenario.get('expected_output'),
                metadata=scenario.get('metadata', {}),
                evaluators=[ToolUsageEvaluator()]
            )
            cases.append(case)
        
        return Dataset(
            cases=cases,
            evaluators=[
                IsInstance(type_name='str'),
                LLMJudge(
                    rubric='Task should be completed using appropriate tools',
                    include_input=True
                )
            ]
        )
    
    def _extract_summary(self, report) -> Dict[str, Any]:
        """Extract summary from evaluation report"""
        # Implementation would extract key metrics from the report
        return {
            'total_cases': len(report.results),
            'success_rate': sum(1 for r in report.results if r.success) / len(report.results),
            'average_score': sum(r.score for r in report.results) / len(report.results)
        }
    
    def _extract_details(self, report) -> List[Dict[str, Any]]:
        """Extract detailed results from evaluation report"""
        # Implementation would extract detailed results
        return [
            {
                'case_name': result.case.name,
                'success': result.success,
                'score': result.score,
                'output': result.output,
                'duration': result.duration
            }
            for result in report.results
        ]
    
    def _create_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparing multiple agents"""
        # Implementation would analyze and compare agent performance
        return {
            'best_agent': max(results.keys(), key=lambda k: results[k]['evaluation_summary']['average_score']),
            'agent_rankings': sorted(
                results.keys(), 
                key=lambda k: results[k]['evaluation_summary']['average_score'], 
                reverse=True
            )
        }
```

## 📊 Evaluation API Endpoints

```python
# src/api/routes/evaluation.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from uuid import UUID

from ...core.database import get_db
from ...core.security import get_current_user
from ...models.user import User
from ...services.evaluation_service import AgentEvaluationService
from ...services.agent_service import AgentService
from ...services.model_service import ModelService
from ...schemas.evaluation import EvaluationRequest, BenchmarkRequest

router = APIRouter()

@router.post("/{agent_id}/evaluate")
async def evaluate_agent(
    agent_id: UUID,
    evaluation_request: EvaluationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Evaluate an agent using provided test cases"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    evaluation_service = AgentEvaluationService(agent_service)
    
    # Create dataset from request
    from pydantic_evals import Dataset, Case
    from pydantic_evals.evaluators import IsInstance, LLMJudge
    
    cases = [
        Case(
            name=f"case_{i}",
            inputs=case['inputs'],
            expected_output=case.get('expected_output'),
            metadata=case.get('metadata', {})
        )
        for i, case in enumerate(evaluation_request.test_cases)
    ]
    
    dataset = Dataset(
        cases=cases,
        evaluators=[
            IsInstance(type_name='str'),
            LLMJudge(rubric=evaluation_request.evaluation_criteria)
        ]
    )
    
    try:
        result = await evaluation_service.evaluate_agent(
            agent_id, dataset, current_user.id, db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

@router.post("/benchmark")
async def benchmark_agents(
    benchmark_request: BenchmarkRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Benchmark multiple agents against the same test cases"""
    
    model_service = ModelService()
    agent_service = AgentService(model_service)
    evaluation_service = AgentEvaluationService(agent_service)
    
    try:
        result = await evaluation_service.create_agent_benchmark(
            benchmark_request.agent_ids,
            benchmark_request.test_cases,
            current_user.id,
            db
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark failed: {str(e)}"
        )

@router.get("/datasets/examples")
async def get_example_datasets():
    """Get example evaluation datasets"""
    
    return {
        "basic_qa": {
            "description": "Basic question-answering evaluation",
            "cases": [
                {
                    "inputs": "What is the capital of France?",
                    "expected_output": "Paris",
                    "metadata": {"difficulty": "easy"}
                }
            ]
        },
        "tool_usage": {
            "description": "Tool usage evaluation",
            "cases": [
                {
                    "inputs": {
                        "task": "Read the file 'data.txt' and count the lines",
                        "expected_tools": ["file_read"]
                    },
                    "expected_output": "File read successfully, X lines found",
                    "metadata": {"category": "file_operations"}
                }
            ]
        }
    }
```

## 📋 Evaluation Schemas

```python
# src/schemas/evaluation.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID

class TestCase(BaseModel):
    """Schema for individual test case"""
    
    name: Optional[str] = None
    inputs: Dict[str, Any]
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = {}

class EvaluationRequest(BaseModel):
    """Schema for evaluation request"""
    
    test_cases: List[TestCase]
    evaluation_criteria: str = "Response should be helpful, accurate, and relevant"
    max_concurrency: Optional[int] = Field(None, gt=0)

class BenchmarkRequest(BaseModel):
    """Schema for benchmark request"""
    
    agent_ids: List[UUID] = Field(..., min_items=2)
    test_cases: List[Dict[str, Any]]
    evaluation_criteria: str = "Response should be helpful, accurate, and relevant"

class EvaluationResult(BaseModel):
    """Schema for evaluation results"""
    
    agent_id: UUID
    total_cases: int
    success_rate: float
    average_score: float
    detailed_results: List[Dict[str, Any]]

class BenchmarkResult(BaseModel):
    """Schema for benchmark results"""
    
    best_agent: UUID
    agent_rankings: List[UUID]
    detailed_results: Dict[UUID, EvaluationResult]
```

## 🔗 Logfire Integration

```python
# src/evaluation/logfire_integration.py
import logfire
from pydantic_evals import Dataset
from typing import Any, Dict

class LogfireEvaluationTracker:
    """Integration with Logfire for evaluation tracking"""
    
    def __init__(self, environment: str = "development"):
        logfire.configure(
            send_to_logfire='if-token-present',
            environment=environment,
            service_name='pydantic-ai-evals'
        )
    
    def track_evaluation(self, dataset: Dataset, task_function: Any) -> Dict[str, Any]:
        """Track evaluation with Logfire"""
        
        with logfire.span('agent_evaluation'):
            # Run evaluation with automatic tracing
            report = dataset.evaluate_sync(task_function)
            
            # Log evaluation summary
            logfire.info(
                'Evaluation completed',
                total_cases=len(report.results),
                success_rate=sum(1 for r in report.results if r.success) / len(report.results),
                average_score=sum(r.score for r in report.results) / len(report.results)
            )
            
            return {
                'report': report,
                'logfire_trace_url': f"https://logfire.pydantic.dev/trace/{logfire.get_current_span().context.trace_id}"
            }
```

## 🎯 Key Features

**🧪 Systematic Evaluation:**
- [`Dataset`](src/evaluation/dataset_manager.py:11) and [`Case`](src/evaluation/dataset_manager.py:13) management
- [`Built-in evaluators`](src/evaluation/evaluators.py:5) (IsInstance, LLMJudge, EqualsExpected)
- [`Custom evaluator creation`](src/evaluation/evaluators.py:8) with scoring logic
- [`Parallel evaluation`](src/services/evaluation_service.py:47) with concurrency control

**📊 Agent Testing:**
- [`Agent performance evaluation`](src/services/evaluation_service.py:26) against test datasets
- [`Multi-agent benchmarking`](src/services/evaluation_service.py:62) and comparison
- [`Tool usage evaluation`](src/evaluation/evaluators.py:26) with span tree analysis
- [`Automated report generation`](src/services/evaluation_service.py:111) with metrics

**🔍 Advanced Analysis:**
- [`OpenTelemetry integration`](src/evaluation/evaluators.py:32) for trace analysis
- [`Logfire visualization`](src/evaluation/logfire_integration.py:12) of evaluation results
- [`Dataset generation`](docs/pydantic_ai_evals.md:1) from LLMs
- [`YAML/JSON export`](docs/pydantic_ai_evals.md:1) for dataset persistence

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*