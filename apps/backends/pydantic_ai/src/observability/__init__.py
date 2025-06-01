"""
Observability module for PydanticAI Agents Service.
"""

from .instrumentation import (
    ObservabilityService,
    observability as observability_service,
    setup_observability,
    get_tracer,
    trace_agent_run,
    trace_model_call,
    trace_database_operation,
    trace_workflow_execution,
    trace_tool_execution,
    add_span_attributes,
    add_span_event,
    record_exception,
)

__all__ = [
    "ObservabilityService",
    "observability_service",
    "setup_observability",
    "get_tracer",
    "trace_agent_run",
    "trace_model_call",
    "trace_database_operation",
    "trace_workflow_execution",
    "trace_tool_execution",
    "add_span_attributes",
    "add_span_event",
    "record_exception",
]