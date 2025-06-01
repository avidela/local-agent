"""
Observability instrumentation using OpenTelemetry and Logfire integration.
"""

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from ..config import settings


class ObservabilityService:
    """Service for managing observability with OpenTelemetry and Logfire."""
    
    def __init__(self):
        """Initialize observability service."""
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self._initialized = False
    
    def setup_instrumentation(self, app=None) -> None:
        """
        Set up OpenTelemetry and Logfire instrumentation.
        
        Args:
            app: FastAPI application instance to instrument
        """
        
        if self._initialized:
            return
        
        try:
            # Try to import and configure Logfire first
            if settings.observability.logfire_token:
                self._setup_logfire(app)
            else:
                # Fallback to standard OpenTelemetry
                self._setup_opentelemetry(app)
            
            # Instrument SQLAlchemy
            self._setup_sqlalchemy_instrumentation()
            
            self._initialized = True
            
        except Exception as e:
            print(f"Failed to setup observability instrumentation: {e}")
    
    def _setup_logfire(self, app=None) -> None:
        """Set up Logfire instrumentation with official integration."""
        
        try:
            import logfire
            
            # Configure Logfire
            logfire.configure(
                token=settings.observability.logfire_token.get_secret_value(),
                service_name=settings.observability.logfire_service_name,
                environment=settings.observability.logfire_environment,
                send_to_logfire='if-token-present',
            )
            
            # Instrument PydanticAI with official integration
            logfire.instrument_pydantic_ai()
            
            # Instrument FastAPI if app provided
            if app:
                logfire.instrument_fastapi(app)
            
            # Instrument SQLAlchemy
            logfire.instrument_sqlalchemy(engine=None)  # Will auto-detect
            
            print("Logfire instrumentation configured successfully")
            
        except ImportError:
            print("Logfire not installed, falling back to OpenTelemetry")
            self._setup_opentelemetry(app)
        except Exception as e:
            print(f"Failed to setup Logfire: {e}")
            self._setup_opentelemetry(app)
    
    def _setup_opentelemetry(self, app=None) -> None:
        """Set up standard OpenTelemetry instrumentation."""
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": settings.observability.otel_service_name,
            "service.version": settings.observability.otel_service_version,
            "environment": settings.observability.logfire_environment,
        })
        
        # Set up tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        
        # Set up span processor and exporter
        if settings.observability.otel_exporter_endpoint:
            # Custom OTLP endpoint
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.observability.otel_exporter_endpoint,
                insecure=True,  # Configure based on your setup
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
        
        # Create tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument FastAPI if app provided
        if app:
            FastAPIInstrumentor.instrument_app(app)
        
        print("OpenTelemetry instrumentation configured successfully")
    
    def _setup_sqlalchemy_instrumentation(self) -> None:
        """Set up SQLAlchemy instrumentation."""
        
        try:
            # Import here to avoid circular dependencies
            from ..database import engine
            
            SQLAlchemyInstrumentor().instrument(
                engine=engine,
                service="pydantic-ai-db",
                enable_commenter=True,
            )
            
        except Exception as e:
            print(f"Failed to instrument SQLAlchemy: {e}")
    
    def get_tracer(self) -> Optional[trace.Tracer]:
        """Get the configured tracer instance."""
        return self.tracer
    
    def trace_agent_run(self, agent_name: str, prompt: str):
        """
        Create a span for agent execution.
        
        Args:
            agent_name: Name of the agent
            prompt: User prompt
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
        
        return self.tracer.start_as_current_span(
            "agent.run",
            attributes={
                "agent.name": agent_name,
                "agent.prompt.length": len(prompt),
                "agent.prompt.preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            }
        )
    
    def trace_model_call(self, model_name: str, tokens: int = 0, cost: float = 0.0):
        """
        Create a span for model API call.
        
        Args:
            model_name: Name of the model
            tokens: Number of tokens used
            cost: Cost of the call
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            from contextlib import nullcontext
            return nullcontext()
        
        return self.tracer.start_as_current_span(
            "model.call",
            attributes={
                "model.name": model_name,
                "model.tokens": tokens,
                "model.cost": cost,
            }
        )
    
    def trace_database_operation(self, operation: str, table: str):
        """
        Create a span for database operation.
        
        Args:
            operation: Type of operation (select, insert, update, delete)
            table: Database table name
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            from contextlib import nullcontext
            return nullcontext()
        
        return self.tracer.start_as_current_span(
            f"db.{operation}",
            attributes={
                "db.operation": operation,
                "db.table": table,
            }
        )
    
    def trace_workflow_execution(self, workflow_name: str, node_name: str):
        """
        Create a span for workflow node execution.
        
        Args:
            workflow_name: Name of the workflow
            node_name: Name of the current node
            
        Returns:
            Span context manager  
        """
        
        if not self.tracer:
            from contextlib import nullcontext
            return nullcontext()
        
        return self.tracer.start_as_current_span(
            "workflow.node",
            attributes={
                "workflow.name": workflow_name,
                "workflow.node": node_name,
            }
        )
    
    def add_span_attributes(self, **attributes) -> None:
        """
        Add attributes to the current span.
        
        Args:
            **attributes: Key-value pairs to add as span attributes
        """
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
    
    def add_span_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """
        Add an event to the current span.
        
        Args:
            name: Event name
            attributes: Optional event attributes
        """
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes or {})
    
    def record_exception(self, exception: Exception) -> None:
        """
        Record an exception in the current span.
        
        Args:
            exception: Exception to record
        """
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.record_exception(exception)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


# Global observability service instance
observability_service = ObservabilityService()


def setup_observability(app=None) -> None:
    """
    Set up observability instrumentation.
    
    Args:
        app: FastAPI application instance
    """
    observability_service.setup_instrumentation(app)


def get_tracer() -> Optional[trace.Tracer]:
    """Get the configured tracer instance."""
    return observability_service.get_tracer()


def trace_agent_run(agent_name: str, prompt: str):
    """Create a span for agent execution."""
    return observability_service.trace_agent_run(agent_name, prompt)


def trace_model_call(model_name: str, tokens: int = 0, cost: float = 0.0):
    """Create a span for model API call."""
    return observability_service.trace_model_call(model_name, tokens, cost)


def trace_database_operation(operation: str, table: str):
    """Create a span for database operation."""
    return observability_service.trace_database_operation(operation, table)


def trace_workflow_execution(workflow_name: str, node_name: str):
    """Create a span for workflow node execution."""
    return observability_service.trace_workflow_execution(workflow_name, node_name)