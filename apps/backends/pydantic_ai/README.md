# PydanticAI Agents Service

A comprehensive, production-ready PydanticAI agents service implementing all official patterns and features.

## Features

### 🔄 Official PydanticAI Integration
- ✅ `GoogleProvider(vertexai=True)` for Gemini models
- ✅ `AnthropicProvider` for Claude models  
- ✅ `agent.run()` and `agent.run_stream()` execution
- ✅ `@agent.tool` and `@agent.tool_plain` decorators
- ✅ `result.all_messages()` and `message_history` parameter

### 💾 Conversation Management
- ✅ Persistent message history with `ModelMessagesTypeAdapter`
- ✅ Streaming responses with `result.stream_text()`
- ✅ Cost tracking with `result.cost()`
- ✅ Session management across multiple turns

### 🎨 Multimodal Input Support
- ✅ Image processing with `ImageUrl` and `BinaryContent`
- ✅ Audio input with `AudioUrl` and binary audio data
- ✅ Video processing with `VideoUrl` and binary video data
- ✅ Document handling with `DocumentUrl` and binary documents

### 🔗 Agent Interoperability
- ✅ Agent2Agent (A2A) Protocol support with `agent.to_a2a()`
- ✅ FastA2A integration for cross-framework communication
- ✅ ASGI-compatible agent servers

### 🧪 Model Evaluation & Testing
- ✅ Pydantic Evals framework for systematic testing
- ✅ Dataset and Case management for evaluation scenarios
- ✅ Built-in evaluators (IsInstance, LLMJudge, EqualsExpected)
- ✅ Custom evaluator creation and scoring

### 🕸️ Workflow Orchestration
- ✅ Graph-based state machines with pydantic-graph
- ✅ Complex workflow modeling and execution
- ✅ State persistence for resumable workflows
- ✅ Human-in-the-loop processes

### 📊 Production Monitoring
- ✅ OpenTelemetry instrumentation with `logfire.instrument_pydantic_ai()`
- ✅ Multiple backend support (Logfire, Jaeger, custom OTel)
- ✅ Performance and cost analytics
- ✅ Error tracking and debugging

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync --dev
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your_key"
   export ANTHROPIC_API_KEY="your_key"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/pydantic_ai"
   ```

3. **Run database migrations:**
   ```bash
   uv run alembic upgrade head
   ```

4. **Start the service:**
   ```bash
   uv run uvicorn src.main:app --reload
   ```

## API Documentation

Once running, visit:
- **Interactive API docs:** http://localhost:8000/docs
- **ReDoc documentation:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

## Architecture

```
src/
├── main.py                    # FastAPI application
├── config/                    # Configuration and settings
├── database/                  # Database models and migrations
├── services/                  # Core business logic
│   ├── agents/               # Agent management
│   ├── models/               # Model providers
│   ├── sessions/             # Session management
│   └── workflows/            # Graph workflows
├── api/                      # FastAPI routes
├── observability/            # Monitoring and tracing
└── utils/                    # Shared utilities

tests/
├── unit/                     # Unit tests with TestModel
├── integration/              # Integration tests
└── workflows/                # Graph workflow tests
```

## Documentation

- **[Complete Implementation Guide](../../../docs/pydantic_ai_index.md)** - Full documentation
- **[Testing Guide](../../../docs/pydantic_ai_testing.md)** - TestModel and FunctionModel patterns
- **[Graphs & Workflows](../../../docs/pydantic_ai_graphs.md)** - State machine workflows
- **[Output Handling](../../../docs/pydantic_ai_output.md)** - Structured outputs and streaming