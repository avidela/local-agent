# 📊 Observability & Monitoring with OpenTelemetry

> **Production-ready monitoring and observability using Pydantic AI's official OpenTelemetry instrumentation**

## 🎯 Overview

Pydantic AI provides built-in OpenTelemetry instrumentation following the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/). This enables comprehensive monitoring with any OpenTelemetry-compatible backend.

## 🚀 Production Setup with Logfire (Recommended)

### ObservabilityService Implementation
```python
# src/observability/instrumentation.py
import os
import logfire
from typing import Optional
from pydantic_ai.agent import Agent, InstrumentationSettings

class ObservabilityService:
    """Service for configuring OpenTelemetry instrumentation for production monitoring"""
    
    def __init__(self, service_name: str = "pydantic-ai-service"):
        self.service_name = service_name
        self._configured = False
    
    def configure_logfire(
        self,
        token: Optional[str] = None,
        send_to_logfire: bool = True,
        event_mode: str = 'events'  # 'events' or 'logs'
    ):
        """Configure Logfire for production monitoring (recommended)"""
        
        if self._configured:
            return
        
        # Configure Logfire with optional token
        if token:
            os.environ['LOGFIRE_TOKEN'] = token
        
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
        
        # Instrument FastAPI for API monitoring
        logfire.instrument_fastapi()
        
        self._configured = True
    
    def configure_alternative_otel_backend(
        self,
        endpoint: str = "http://localhost:4318",
        event_mode: str = 'events'
    ):
        """Configure Logfire SDK with alternative OpenTelemetry backend"""
        
        if self._configured:
            return
        
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
        logfire.instrument_fastapi()
        
        self._configured = True
    
    def configure_raw_opentelemetry(
        self,
        endpoint: str = "http://localhost:4318"
    ):
        """Configure raw OpenTelemetry without Logfire"""
        
        if self._configured:
            return
        
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
        
        # Create custom instrumentation settings
        instrumentation_settings = InstrumentationSettings(
            tracer_provider=tracer_provider,
            event_logger_provider=event_logger_provider,
            include_binary_content=False  # Exclude binary for performance
        )
        
        # Instrument all agents
        Agent.instrument_all(instrumentation_settings)
        
        self._configured = True
    
    def get_custom_instrumentation_settings(
        self,
        include_binary_content: bool = False,
        event_mode: str = 'events'
    ) -> InstrumentationSettings:
        """Get custom instrumentation settings for specific agents"""
        
        return InstrumentationSettings(
            include_binary_content=include_binary_content
        )

# Global observability service instance
observability = ObservabilityService()
```

## 🔧 Enhanced Model Service with Instrumentation

```python
# src/services/model_service.py - Enhanced with observability
from pydantic_ai.models.instrumented import InstrumentedModel
from typing import Union
from ..observability.instrumentation import observability

class ModelService:
    """Enhanced model service with OpenTelemetry instrumentation"""
    
    def get_model(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        instrumented: bool = True
    ) -> Union[GoogleModel, AnthropicModel, InstrumentedModel]:
        """Get model with optional instrumentation for observability"""
        
        # Get base model (Google/Anthropic)
        base_model = self._create_base_model(provider, model_name, temperature, max_tokens)
        
        # Wrap with instrumentation for observability
        if instrumented:
            instrumentation_settings = observability.get_custom_instrumentation_settings(
                include_binary_content=False
            )
            return InstrumentedModel(base_model, instrumentation_settings)
        
        return base_model
```

## 🏗️ FastAPI Application with Observability

```python
# src/main.py - Enhanced with monitoring
from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
from .observability.instrumentation import observability

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with observability setup"""
    
    # Initialize observability based on environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Production: Use Logfire platform
        observability.configure_logfire(
            token=os.getenv("LOGFIRE_TOKEN"),
            send_to_logfire=True,
            event_mode='events'
        )
    elif environment == "staging":
        # Staging: Use alternative OTel backend
        observability.configure_alternative_otel_backend(
            endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4318"),
            event_mode='logs'  # Better for long conversations
        )
    else:
        # Development: Use raw OpenTelemetry or local setup
        if os.getenv("OTEL_ENDPOINT"):
            observability.configure_raw_opentelemetry(
                endpoint=os.getenv("OTEL_ENDPOINT")
            )
    
    yield

def create_app() -> FastAPI:
    """Create FastAPI application with observability"""
    
    app = FastAPI(
        title="Pydantic AI Agents Service",
        description="Production-ready AI agents service with observability",
        version="1.0.0",
        lifespan=lifespan
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint with tracing"""
        return {"status": "healthy", "service": "pydantic-ai-agents"}
    
    return app
```

## 🐳 Docker Setup with OpenTelemetry Collector

```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  # PydanticAI Service
  pydantic-ai-service:
    build: .
    environment:
      - ENVIRONMENT=staging
      - OTEL_ENDPOINT=http://otel-collector:4318
    depends_on:
      - otel-collector
    ports:
      - "8000:8000"

  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./observability/otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
    depends_on:
      - jaeger

  # Jaeger for trace visualization
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # Jaeger gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  # Alternative: OTEL TUI for development
  otel-tui:
    image: ymtdzzz/otel-tui:latest
    ports:
      - "4318:4318"
    profiles:
      - dev
```

## ⚙️ OpenTelemetry Collector Configuration

```yaml
# observability/otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
  memory_limiter:
    limit_mib: 512

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  # Optional: Export to Logfire
  otlphttp:
    endpoint: https://logfire-api.pydantic.dev
    headers:
      authorization: "Bearer ${LOGFIRE_TOKEN}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger]
```

## 📊 Custom Metrics Collection

```python
# src/observability/metrics.py
import logfire
from typing import Dict, Any
import time

class MetricsCollector:
    """Custom metrics collection for production monitoring"""
    
    def track_agent_execution(self, agent_id: str, user_id: str):
        """Track agent execution metrics"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Log successful execution
                    logfire.info(
                        "Agent execution completed",
                        agent_id=agent_id,
                        user_id=user_id,
                        execution_time=time.time() - start_time,
                        status="success"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log failed execution
                    logfire.error(
                        "Agent execution failed",
                        agent_id=agent_id,
                        user_id=user_id,
                        execution_time=time.time() - start_time,
                        error=str(e),
                        status="error"
                    )
                    raise
            
            return wrapper
        return decorator
    
    def track_model_usage(self, model_name: str, provider: str):
        """Track model usage and costs"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                # Extract usage information from result
                if hasattr(result, 'cost') and hasattr(result, 'all_messages'):
                    logfire.info(
                        "Model usage tracked",
                        model_name=model_name,
                        provider=provider,
                        cost=result.cost(),
                        message_count=len(result.all_messages()),
                        total_tokens=getattr(result, 'usage', {}).get('total_tokens', 0)
                    )
                
                return result
            
            return wrapper
        return decorator

# Global metrics collector
metrics = MetricsCollector()
```

## 🌐 Environment Configuration

```bash
# .env.production
ENVIRONMENT=production
LOGFIRE_TOKEN=your_logfire_token_here

# .env.staging  
ENVIRONMENT=staging
OTEL_ENDPOINT=http://your-otel-collector:4318

# .env.development
ENVIRONMENT=development
OTEL_ENDPOINT=http://localhost:4318
```

## 🔧 Configuration Options

### Event Mode Configuration
- **`event_mode='events'`**: Default mode, efficient for most use cases
- **`event_mode='logs'`**: Better for long conversations to avoid truncation

### Instrumentation Settings
- **`include_binary_content=False`**: Exclude binary content for performance
- **Custom providers**: Set custom `tracer_provider` and `event_logger_provider`

### Backend Options
1. **Logfire Platform** (Recommended): Best experience with Pydantic AI
2. **Alternative OTel**: Jaeger, Zipkin, or other OTel-compatible backends
3. **Raw OpenTelemetry**: Custom OTel setup without Logfire SDK

## 📈 Key Monitoring Features

**🔍 Automatic Instrumentation:**
- [`logfire.instrument_pydantic_ai()`](https://logfire.pydantic.dev/) - Agent execution tracing
- [`logfire.instrument_httpx()`](https://logfire.pydantic.dev/) - Model API call monitoring
- [`logfire.instrument_sqlalchemy()`](https://logfire.pydantic.dev/) - Database operation tracking
- [`logfire.instrument_fastapi()`](https://logfire.pydantic.dev/) - API endpoint monitoring

**📊 Cost & Usage Tracking:**
- Model usage with [`result.cost()`](https://docs.pydantic.ai/api/agents/#pydantic_ai.RunResult.cost)
- Token consumption and message counts
- Performance metrics and execution times
- Error rates and debugging information

**🔧 Advanced Configuration:**
- [`InstrumentationSettings`](https://docs.pydantic.ai/agents/#pydantic_ai.InstrumentationSettings) for custom setup
- [`Agent.instrument_all()`](https://docs.pydantic.ai/agents/#pydantic_ai.Agent.instrument_all) for global instrumentation
- [`InstrumentedModel`](https://docs.pydantic.ai/models/instrumented/) for model-specific monitoring

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*