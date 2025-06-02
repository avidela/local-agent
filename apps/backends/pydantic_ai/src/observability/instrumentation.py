"""
Observability instrumentation using OpenTelemetry and Logfire integration.
"""

import os
from typing import Optional, Literal
from contextlib import nullcontext

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
    
    def __init__(self, service_name: str = "pydantic-ai-service"):
        """Initialize observability service."""
        self.service_name = service_name
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None
        self._configured = False
    
    def configure_logfire(
        self,
        token: Optional[str] = None,
        send_to_logfire: bool = True,
        event_mode: Literal['attributes', 'logs'] = 'attributes'
    ):
        """Configure Logfire for production monitoring (recommended)"""
        
        if self._configured:
            return
        
        try:
            import logfire
            
            # Configure Logfire with optional token
            if token:
                os.environ['LOGFIRE_TOKEN'] = token
            elif settings.observability.logfire_token:
                os.environ['LOGFIRE_TOKEN'] = settings.observability.logfire_token.get_secret_value()
            
            logfire.configure(
                service_name=self.service_name,
                send_to_logfire=send_to_logfire
            )
            
            # Instrument PydanticAI with semantic conventions
            logfire.instrument_pydantic_ai(event_mode=event_mode)
            
            # Instrument HTTP requests for model API calls
            logfire.instrument_httpx(capture_all=True)
            
            # Instrument SQLAlchemy for database operations
            logfire.instrument_sqlalchemy()
            
            # Note: FastAPI instrumentation will be done during app setup
            
            self._configured = True
            print("Logfire instrumentation configured successfully")
            
        except ImportError:
            print("Logfire not installed, falling back to OpenTelemetry")
            self.configure_raw_opentelemetry()
        except Exception as e:
            print(f"Failed to setup Logfire: {e}")
            self.configure_raw_opentelemetry()
    
    def configure_alternative_otel_backend(
        self,
        endpoint: str = "http://localhost:4318",
        event_mode: Literal['attributes', 'logs'] = 'attributes'
    ):
        """Configure Logfire SDK with alternative OpenTelemetry backend"""
        
        if self._configured:
            return
        
        try:
            import logfire
            
            # Set OTel endpoint
            os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = endpoint
            
            # Configure Logfire to send to alternative backend
            logfire.configure(
                service_name=self.service_name,
                send_to_logfire=False  # Don't send to Logfire platform
            )
            
            # Instrument components
            logfire.instrument_pydantic_ai(event_mode=event_mode)
            logfire.instrument_httpx(capture_all=True)
            logfire.instrument_sqlalchemy()
            
            # Note: FastAPI instrumentation will be done during app setup
            
            self._configured = True
            print(f"Alternative OpenTelemetry backend configured: {endpoint}")
            
        except ImportError:
            print("Logfire not installed, using raw OpenTelemetry")
            self.configure_raw_opentelemetry(endpoint)
        except Exception as e:
            print(f"Failed to setup alternative backend: {e}")
            self.configure_raw_opentelemetry(endpoint)
    
    def configure_raw_opentelemetry(
        self,
        endpoint: str = "http://localhost:4318"
    ):
        """Configure raw OpenTelemetry without Logfire"""
        
        if self._configured:
            return
        
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.trace import set_tracer_provider
            from opentelemetry.sdk._events import EventLoggerProvider
            
            # Configure OTel endpoint
            os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = endpoint
            
            # Setup trace provider
            exporter = OTLPSpanExporter()
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(span_processor)
            set_tracer_provider(tracer_provider)
            
            # Setup event logger provider
            event_logger_provider = EventLoggerProvider()
            
            # Create tracer
            self.tracer = trace.get_tracer(__name__)
            
            self._configured = True
            print(f"Raw OpenTelemetry configured: {endpoint}")
            
        except Exception as e:
            print(f"Failed to setup raw OpenTelemetry: {e}")
            self._configured = True  # Mark as configured to prevent retries
    
    def setup_instrumentation(self, app=None) -> None:
        """
        Set up OpenTelemetry and Logfire instrumentation.
        
        Args:
            app: FastAPI application instance to instrument
        """
        
        if self._configured:
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
            
            self._configured = True
            
        except Exception as e:
            print(f"Failed to setup observability instrumentation: {e}")
    
    def _setup_logfire(self, app=None) -> None:
        """Set up Logfire instrumentation with official integration."""
        
        try:
            import logfire
            
            # Configure Logfire
            token = None
            if settings.observability.logfire_token:
                token = settings.observability.logfire_token.get_secret_value()
            
            logfire.configure(
                token=token,
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
    
    def trace_agent_run(self, agent_name: str, prompt: str, **kwargs):
        """
        Create a span for agent execution with enhanced attributes.
        
        Args:
            agent_name: Name of the agent
            prompt: User prompt
            **kwargs: Additional attributes (user_id, session_id, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "agent.name": agent_name,
            "agent.prompt.length": len(prompt),
            "agent.prompt.preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        }
        
        # Add optional attributes
        for key, value in kwargs.items():
            if value is not None:
                attributes[f"agent.{key}"] = str(value)
        
        return self.tracer.start_as_current_span(
            "agent.run",
            attributes=attributes
        )
    
    def trace_model_call(self, model_name: str, tokens: int = 0, cost: float = 0.0, **kwargs):
        """
        Create a span for model API call with enhanced metrics.
        
        Args:
            model_name: Name of the model
            tokens: Number of tokens used
            cost: Cost of the call
            **kwargs: Additional attributes (input_tokens, output_tokens, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "model.name": model_name,
            "model.tokens.total": tokens,
            "model.cost": cost,
        }
        
        # Add optional token breakdown
        if "input_tokens" in kwargs:
            attributes["model.tokens.input"] = kwargs["input_tokens"]
        if "output_tokens" in kwargs:
            attributes["model.tokens.output"] = kwargs["output_tokens"]
        if "provider" in kwargs:
            attributes["model.provider"] = kwargs["provider"]
        
        return self.tracer.start_as_current_span(
            "model.call",
            attributes=attributes
        )
    
    def trace_database_operation(self, operation: str, table: str, **kwargs):
        """
        Create a span for database operation with enhanced attributes.
        
        Args:
            operation: Type of operation (select, insert, update, delete)
            table: Database table name
            **kwargs: Additional attributes (user_id, record_count, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "db.operation": operation,
            "db.table": table,
            "db.system": "postgresql",
        }
        
        # Add optional attributes
        for key, value in kwargs.items():
            if value is not None:
                attributes[f"db.{key}"] = str(value)
        
        return self.tracer.start_as_current_span(
            f"db.{operation}",
            attributes=attributes
        )
    
    def trace_workflow_execution(self, workflow_name: str, node_name: str, **kwargs):
        """
        Create a span for workflow node execution with enhanced attributes.
        
        Args:
            workflow_name: Name of the workflow
            node_name: Name of the current node
            **kwargs: Additional attributes (session_id, user_id, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "workflow.name": workflow_name,
            "workflow.node": node_name,
        }
        
        # Add optional attributes
        for key, value in kwargs.items():
            if value is not None:
                attributes[f"workflow.{key}"] = str(value)
        
        return self.tracer.start_as_current_span(
            "workflow.node",
            attributes=attributes
        )
    
    def trace_tool_execution(self, tool_name: str, operation: str, **kwargs):
        """
        Create a span for tool execution with performance metrics.
        
        Args:
            tool_name: Name of the tool being executed
            operation: Operation being performed
            **kwargs: Additional attributes (file_path, search_query, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "tool.name": tool_name,
            "tool.operation": operation,
        }
        
        # Add operation-specific attributes
        for key, value in kwargs.items():
            if value is not None:
                attributes[f"tool.{key}"] = str(value)
        
        return self.tracer.start_as_current_span(
            f"tool.{operation}",
            attributes=attributes
        )
    
    def trace_multimodal_operation(self, operation: str, **kwargs):
        """
        Create a span for multimodal file operations.
        
        Args:
            operation: Operation being performed (upload_file, process_file, etc.)
            **kwargs: Additional attributes (filename, content_type, file_id, etc.)
            
        Returns:
            Span context manager
        """
        
        if not self.tracer:
            return nullcontext()
        
        attributes = {
            "multimodal.operation": operation,
        }
        
        # Add operation-specific attributes
        for key, value in kwargs.items():
            if value is not None:
                attributes[f"multimodal.{key}"] = str(value)
        
        return self.tracer.start_as_current_span(
            f"multimodal.{operation}",
            attributes=attributes
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
observability = ObservabilityService()


def setup_observability(app=None) -> None:
    """
    Set up observability instrumentation with environment-based configuration.
    
    Args:
        app: FastAPI application instance
    """
    
    # Environment-based configuration
    environment = settings.observability.logfire_environment
    
    if environment == "production":
        # Production: Use Logfire platform with provided token from environment
        token = settings.observability.logfire_token
        if not token:
            raise ValueError("LOGFIRE_TOKEN environment variable must be set for production")
        
        observability.configure_logfire(
            token=token.get_secret_value(),
            send_to_logfire=True,
            event_mode='attributes'
        )
    elif environment == "staging":
        # Staging: Use alternative OTel backend
        observability.configure_alternative_otel_backend(
            endpoint=settings.observability.otel_exporter_endpoint or "http://localhost:4318",
            event_mode='logs'  # Better for long conversations
        )
    else:
        # Development: Use raw OpenTelemetry or fallback
        if settings.observability.otel_exporter_endpoint:
            observability.configure_raw_opentelemetry(
                endpoint=settings.observability.otel_exporter_endpoint
            )
        elif settings.observability.logfire_token:
            # Use Logfire token if available in development
            token = settings.observability.logfire_token
            observability.configure_logfire(
                token=token.get_secret_value(),
                send_to_logfire=True,
                event_mode='attributes'
            )
    
    # Instrument FastAPI if app provided
    if app and observability._configured:
        try:
            import logfire
            logfire.instrument_fastapi(app)
            print("FastAPI instrumentation configured")
        except ImportError:
            print("Logfire not available for FastAPI instrumentation")
        except Exception as e:
            print(f"Failed to instrument FastAPI: {e}")


def trace_multimodal_operation(operation: str, **kwargs):
    """Get multimodal operation tracer."""
    return observability.trace_multimodal_operation(operation, **kwargs)


def get_tracer() -> Optional[trace.Tracer]:
    """Get the configured tracer instance."""
    return observability.get_tracer()


def trace_agent_run(agent_name: str, prompt: str, **kwargs):
    """Create a span for agent execution."""
    return observability.trace_agent_run(agent_name, prompt, **kwargs)


def trace_model_call(model_name: str, tokens: int = 0, cost: float = 0.0, **kwargs):
    """Create a span for model API call."""
    return observability.trace_model_call(model_name, tokens, cost, **kwargs)


def trace_database_operation(operation: str, table: str, **kwargs):
    """Create a span for database operation."""
    return observability.trace_database_operation(operation, table, **kwargs)


def trace_workflow_execution(workflow_name: str, node_name: str, **kwargs):
    """Create a span for workflow node execution."""
    return observability.trace_workflow_execution(workflow_name, node_name, **kwargs)


def trace_tool_execution(tool_name: str, operation: str, **kwargs):
    """Create a span for tool execution."""
    return observability.trace_tool_execution(tool_name, operation, **kwargs)


def add_span_attributes(**attributes) -> None:
    """Add attributes to the current span."""
    observability.add_span_attributes(**attributes)


def add_span_event(name: str, attributes: Optional[dict] = None) -> None:
    """Add an event to the current span."""
    observability.add_span_event(name, attributes)


def record_exception(exception: Exception) -> None:
    """Record an exception in the current span."""
    observability.record_exception(exception)