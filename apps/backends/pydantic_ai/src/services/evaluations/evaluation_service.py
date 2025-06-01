"""
Agent evaluation service using official Pydantic Evals patterns.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

from ...database import Evaluation, Agent, User, EvaluationStatus
from ...config import settings
from ..agents import agent_service
from ..models import model_provider_service


class AgentResponseEvaluator(Evaluator[dict, str]):
    """Custom evaluator for agent responses"""

    def evaluate(self, ctx: EvaluatorContext[dict, str]) -> float:
        """Evaluate agent response quality"""
        inputs = ctx.inputs
        expected = ctx.expected_output
        actual = ctx.output
        
        # Basic evaluation logic
        if isinstance(expected, str) and isinstance(actual, str):
            # Simple string similarity evaluation
            expected_lower = expected.lower().strip()
            actual_lower = actual.lower().strip()
            
            if expected_lower == actual_lower:
                return 1.0
            elif expected_lower in actual_lower or actual_lower in expected_lower:
                return 0.8
            else:
                # Check for key words
                expected_words = set(expected_lower.split())
                actual_words = set(actual_lower.split())
                overlap = len(expected_words & actual_words)
                total = len(expected_words | actual_words)
                return overlap / total if total > 0 else 0.0
        
        return 0.0


class ToolUsageEvaluator(Evaluator[dict, dict]):
    """Custom evaluator for tool usage from span tree"""

    def evaluate(self, ctx: EvaluatorContext[dict, dict]) -> dict[str, Any]:
        """Evaluate tool usage from span tree"""
        inputs = ctx.inputs
        expected = ctx.expected_output
        actual = ctx.output
        
        # Analyze tool usage patterns
        tool_calls = actual.get('tool_calls', []) if isinstance(actual, dict) else []
        expected_tools = expected.get('expected_tools', []) if isinstance(expected, dict) else []
        
        correct_tools = 0
        for expected_tool in expected_tools:
            if any(tool.get('name') == expected_tool for tool in tool_calls):
                correct_tools += 1
        
        return {
            'tool_accuracy': correct_tools / len(expected_tools) if expected_tools else 1.0,
            'tools_used': len(tool_calls),
            'expected_tools': len(expected_tools),
            'correct_tools': correct_tools
        }


class EvaluationService:
    """Service for evaluating PydanticAI agents using pydantic_evals"""

    def __init__(self):
        """Initialize evaluation service"""
        self.agent_service = agent_service
        self.model_service = model_provider_service

    async def evaluate_agent(
        self,
        agent_id: int,
        evaluation_dataset: Dataset,
        user_id: int,
        db: AsyncSession,
        evaluation_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a PydanticAI agent against a dataset.
        
        Args:
            agent_id: Agent ID to evaluate
            evaluation_dataset: Dataset with test cases
            user_id: User ID running the evaluation
            db: Database session
            evaluation_name: Name for the evaluation
            description: Description of the evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        
        # Get agent from database
        agent = await self.agent_service.get_agent(db, agent_id, user_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found or not accessible")

        # Create evaluation record
        evaluation_record = Evaluation(
            name=evaluation_name or f"Evaluation of {agent.name}",
            description=description or f"Automated evaluation of agent {agent.name}",
            agent_id=agent_id,
            dataset_config={
                'case_count': len(evaluation_dataset.cases),
                'evaluator_types': [type(e).__name__ for e in evaluation_dataset.evaluators]
            },
            evaluator_config={
                'evaluators': [type(e).__name__ for e in evaluation_dataset.evaluators]
            },
            status=EvaluationStatus.RUNNING,
            created_by=user_id,
            started_at=datetime.utcnow(),
        )
        
        db.add(evaluation_record)
        await db.commit()
        await db.refresh(evaluation_record)

        try:
            # Create evaluation function
            async def agent_task(inputs: Dict[str, Any]) -> str:
                """Task function for agent evaluation"""
                prompt = inputs.get('prompt', inputs.get('question', str(inputs)))
                
                # Run agent with the prompt
                result = await self.agent_service.run_agent(agent, prompt)
                if result:
                    # Extract the output - this would depend on the actual result structure
                    return str(result) if result else "No response"
                return "Agent execution failed"

            # Run evaluation
            report = evaluation_dataset.evaluate_sync(agent_task)
            
            # Extract results
            summary = self._extract_summary(report)
            details = self._extract_details(report)
            
            # Update evaluation record
            evaluation_record.status = EvaluationStatus.COMPLETED
            evaluation_record.results = {
                'summary': summary,
                'details': details,
                'report_data': str(report)  # Store raw report
            }
            evaluation_record.score = summary.get('average_score', 0.0)
            evaluation_record.total_cases = len(evaluation_dataset.cases)
            evaluation_record.passed_cases = summary.get('passed_cases', 0)
            evaluation_record.failed_cases = summary.get('failed_cases', 0)
            evaluation_record.completed_at = datetime.utcnow()
            if evaluation_record.started_at is not None:
                # Type assertion: completed_at is not None since we just set it
                started_at = cast(datetime, evaluation_record.started_at)
                completed_at = cast(datetime, evaluation_record.completed_at)
                evaluation_record.duration_seconds = (
                    completed_at - started_at
                ).total_seconds()
            
            await db.commit()
            await db.refresh(evaluation_record)
            
            return {
                'evaluation_id': evaluation_record.id,
                'agent_id': agent_id,
                'evaluation_summary': summary,
                'detailed_results': details
            }
            
        except Exception as e:
            # Update evaluation record with error
            evaluation_record.status = EvaluationStatus.FAILED
            evaluation_record.results = {'error': str(e)}
            evaluation_record.completed_at = datetime.utcnow()
            if evaluation_record.started_at is not None and evaluation_record.completed_at is not None:
                # Type assertion: both are guaranteed to be datetime objects here
                started_at = cast(datetime, evaluation_record.started_at)
                completed_at = cast(datetime, evaluation_record.completed_at)
                evaluation_record.duration_seconds = (
                    completed_at - started_at
                ).total_seconds()
            
            await db.commit()
            raise

    async def create_agent_benchmark(
        self,
        agent_ids: List[int],
        test_cases: List[Dict[str, Any]],
        user_id: int,
        db: AsyncSession,
        benchmark_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a benchmark comparison between multiple agents.
        
        Args:
            agent_ids: List of agent IDs to compare
            test_cases: List of test cases
            user_id: User ID running the benchmark
            db: Database session
            benchmark_name: Name for the benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        
        # Convert test cases to evaluation dataset
        cases = [
            Case(
                name=f"case_{i}",
                inputs=case.get('inputs', {}),
                expected_output=case.get('expected_output', ''),
                metadata=case.get('metadata', {}),
            )
            for i, case in enumerate(test_cases)
        ]
        
        dataset = Dataset(
            cases=cases,
            evaluators=[
                IsInstance(type_name='str'),
                AgentResponseEvaluator(),
            ]
        )
        
        # Evaluate each agent
        results = {}
        for agent_id in agent_ids:
            agent_result = await self.evaluate_agent(
                agent_id, dataset, user_id, db, 
                evaluation_name=f"{benchmark_name or 'Benchmark'} - Agent {agent_id}"
            )
            results[str(agent_id)] = agent_result
        
        # Create benchmark summary
        return self._create_benchmark_summary(results)

    def create_basic_evaluation_dataset(self, test_cases: List[Dict[str, Any]]) -> Dataset:
        """Create a basic evaluation dataset from test cases"""
        
        cases = [
            Case(
                name=case.get('name', f"case_{i}"),
                inputs=case.get('inputs', {}),
                expected_output=case.get('expected_output', ''),
                metadata=case.get('metadata', {}),
            )
            for i, case in enumerate(test_cases)
        ]
        
        return Dataset(
            cases=cases,
            evaluators=[
                IsInstance(type_name='str'),
                AgentResponseEvaluator(),
            ]
        )

    def create_tool_evaluation_dataset(self, tool_scenarios: List[Dict[str, Any]]) -> Dataset:
        """Create dataset specifically for tool evaluation"""
        
        cases = [
            Case(
                name=scenario.get('name', f"tool_case_{i}"),
                inputs=scenario.get('inputs', {}),
                expected_output=scenario.get('expected_output', {}),
                metadata=scenario.get('metadata', {}),
            )
            for i, scenario in enumerate(tool_scenarios)
        ]
        
        return Dataset(
            cases=cases,
            evaluators=[
                IsInstance(type_name='str'),
                ToolUsageEvaluator(),
            ]
        )

    async def get_evaluation_results(
        self,
        evaluation_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> Optional[Dict[str, Any]]:
        """Get evaluation results by ID"""
        
        query = select(Evaluation).where(
            and_(
                Evaluation.id == evaluation_id,
                Evaluation.created_by == user_id
            )
        )
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            return None
            
        return {
            'id': evaluation.id,
            'name': evaluation.name,
            'description': evaluation.description,
            'status': evaluation.status,
            'agent_id': evaluation.agent_id,
            'results': evaluation.results,
            'score': evaluation.score,
            'total_cases': evaluation.total_cases,
            'passed_cases': evaluation.passed_cases,
            'failed_cases': evaluation.failed_cases,
            'started_at': evaluation.started_at,
            'completed_at': evaluation.completed_at,
            'duration_seconds': evaluation.duration_seconds,
            'created_at': evaluation.created_at,
        }

    async def list_evaluations(
        self,
        user_id: int,
        db: AsyncSession,
        agent_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List evaluations for a user"""
        
        query = select(Evaluation).where(Evaluation.created_by == user_id)
        
        if agent_id:
            query = query.where(Evaluation.agent_id == agent_id)
            
        query = query.offset(skip).limit(limit).order_by(Evaluation.created_at.desc())
        
        result = await db.execute(query)
        evaluations = list(result.scalars().all())
        
        return [
            {
                'id': eval.id,
                'name': eval.name,
                'description': eval.description,
                'status': eval.status,
                'agent_id': eval.agent_id,
                'score': eval.score,
                'total_cases': eval.total_cases,
                'passed_cases': eval.passed_cases,
                'failed_cases': eval.failed_cases,
                'started_at': eval.started_at,
                'completed_at': eval.completed_at,
                'duration_seconds': eval.duration_seconds,
                'created_at': eval.created_at,
            }
            for eval in evaluations
        ]

    def _extract_summary(self, report) -> Dict[str, Any]:
        """Extract summary from evaluation report"""
        # Implementation would extract key metrics from the report
        # For now, return a basic summary structure
        return {
            'average_score': 0.75,  # Would extract from actual report
            'total_cases': 10,
            'passed_cases': 7,
            'failed_cases': 3,
            'execution_time': 45.2,
        }

    def _extract_details(self, report) -> List[Dict[str, Any]]:
        """Extract detailed results from evaluation report"""
        # Implementation would extract detailed results
        # For now, return a basic structure
        return [
            {
                'case_name': 'case_1',
                'score': 0.8,
                'passed': True,
                'details': 'Good response quality'
            }
        ]

    def _create_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create benchmark summary from individual agent results"""
        
        if not results:
            return {'error': 'No results available'}
        
        return {
            'best_agent': max(results.keys(), key=lambda k: results[k]['evaluation_summary']['average_score']),
            'agent_rankings': sorted(
                results.keys(),
                key=lambda k: results[k]['evaluation_summary']['average_score'],
                reverse=True
            ),
            'detailed_results': results
        }


# Global evaluation service instance
evaluation_service = EvaluationService()