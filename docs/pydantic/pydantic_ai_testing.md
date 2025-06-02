# PydanticAI Unit Testing Guide

## Overview

This guide covers comprehensive unit testing strategies for PydanticAI applications, focusing on practical patterns that avoid the costs, latency, and variability of real LLM calls during testing.

## Testing Strategy

### Recommended Testing Stack

For effective PydanticAI testing, follow this established pattern:

1. **Use [`pytest`](https://pytest.org/) as your test harness** - Industry standard for Python testing
2. **Use [`inline-snapshot`](https://inline-snapshot.readthedocs.io/) for long assertions** - Reduces verbose test code
3. **Use [`dirty-equals`](https://dirty-equals.readthedocs.io/) for complex data structures** - Flexible equality comparisons
4. **Use [`TestModel`](src/models/test_model.py) or [`FunctionModel`](src/models/function_model.py) instead of real models** - Eliminate LLM API calls
5. **Use [`Agent.override()`](src/agents/agent.py) to replace models in application logic** - Clean dependency injection
6. **Set `ALLOW_MODEL_REQUESTS=False` globally** - Prevent accidental real API calls

### Core Testing Principles

```python
# Global test configuration
import pytest
from pydantic_ai import models

# Prevent accidental real model requests
models.ALLOW_MODEL_REQUESTS = False

# Use anyio for async tests
pytestmark = pytest.mark.anyio
```

## Testing with TestModel

### What is TestModel?

[`TestModel`](src/models/test_model.py) is a test-friendly model implementation that:

- **Automatically generates valid structured data** based on tool schemas
- **Calls all registered agent tools** during test execution
- **Returns appropriate responses** (plain text or structured) based on agent return types
- **Uses procedural Python code** (no ML/AI) to satisfy JSON schemas

> **Important**: TestModel is not magic - it generates schema-compliant but not semantically meaningful data. For more sophisticated testing, use [`FunctionModel`](src/models/function_model.py).

### Example Application Code

```python
# weather_app.py
import asyncio
from datetime import date
from pydantic_ai import Agent, RunContext
from fake_database import DatabaseConn
from weather_service import WeatherService

weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)

async def run_weather_forecast(
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:
        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)
        
        # Run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
```

### TestModel Unit Tests

```python
# test_weather_app.py
from datetime import timezone
import pytest
from dirty_equals import IsNow, IsStr
from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelResponse, SystemPromptPart, TextPart, ToolCallPart,
    ToolReturnPart, UserPromptPart, ModelRequest,
)
from pydantic_ai.usage import Usage
from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)
    
    # Verify stored forecast
    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'
    
    # Verify complete message flow
    assert messages == [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=Usage(
                requests=1, request_tokens=71, response_tokens=7,
                total_tokens=78, details=None,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=Usage(
                requests=1, request_tokens=77, response_tokens=16,
                total_tokens=93, details=None,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]
```

## Testing with FunctionModel

### Advanced Control with FunctionModel

[`FunctionModel`](src/models/function_model.py) provides complete control over model responses, allowing you to:

- **Test specific code paths** with custom tool arguments
- **Simulate different model behaviors** and edge cases
- **Create deterministic test scenarios** with precise inputs

### Custom Function Model Example

```python
# test_weather_app2.py
import re
import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, TextPart, ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

def call_weather_forecast(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # First call: extract date and call weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # Second call: return the forecast result
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])

async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)
    
    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'
```

## Pytest Fixtures for Model Override

### Reusable Test Setup

For testing suites with many tests requiring model overrides, use pytest fixtures:

```python
# tests.py
import pytest
from weather_app import weather_agent
from pydantic_ai.models.test import TestModel

@pytest.fixture
def override_weather_agent():
    """Override weather agent with TestModel for all tests."""
    with weather_agent.override(model=TestModel()):
        yield

async def test_forecast(override_weather_agent: None):
    # Test code here - model is automatically overridden
    pass

@pytest.fixture
def override_with_function_model():
    """Override with custom FunctionModel for specific scenarios."""
    def custom_model_function(messages, info):
        # Custom model logic here
        pass
    
    with weather_agent.override(model=FunctionModel(custom_model_function)):
        yield
```

## Testing Patterns and Best Practices

### Message Capture and Validation

```python
from pydantic_ai import capture_run_messages

async def test_conversation_flow():
    with capture_run_messages() as messages:
        with agent.override(model=TestModel()):
            result = await agent.run("Test prompt")
    
    # Validate complete conversation
    assert len(messages) == 4  # Request -> Response -> Request -> Response
    assert messages[0].parts[0].content == "System prompt content"
    assert messages[1].parts[0].tool_name == "expected_tool"
```

### Testing Different Model Providers

```python
@pytest.mark.parametrize("model_provider", [
    TestModel(),
    FunctionModel(custom_function),
])
async def test_with_different_models(model_provider):
    with agent.override(model=model_provider):
        result = await agent.run("Test prompt")
        assert result.output is not None
```

### Testing Async Dependencies

```python
async def test_with_async_deps():
    async with WeatherService() as weather_service:
        with weather_agent.override(model=TestModel()):
            result = await weather_agent.run(
                "Weather in London", 
                deps=weather_service
            )
            assert "weather" in result.output.lower()
```

### Error Handling Tests

```python
def failing_model_function(messages, info):
    raise ValueError("Simulated model failure")

async def test_model_error_handling():
    with pytest.raises(ValueError, match="Simulated model failure"):
        with agent.override(model=FunctionModel(failing_model_function)):
            await agent.run("Test prompt")
```

## Integration with Testing Frameworks

### Pytest Configuration

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Environment variables for testing
env = [
    "ALLOW_MODEL_REQUESTS=False",
    "PYDANTIC_AI_TEST_MODE=True",
]
```

### Coverage Configuration

```python
# .coveragerc
[run]
source = src/
omit = 
    */tests/*
    */test_*.py
    
[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Testing Checklist

### Essential Test Coverage

- [ ] **Agent tool calls** with various inputs
- [ ] **Conversation flows** with message history
- [ ] **Dependency injection** with different contexts  
- [ ] **Error handling** for model failures
- [ ] **Streaming responses** if used
- [ ] **Multi-turn conversations** 
- [ ] **Tool argument validation**
- [ ] **Output format verification**

### Performance Testing

- [ ] **Parallel execution** with `asyncio.gather()`
- [ ] **Resource cleanup** in async contexts
- [ ] **Memory usage** with large conversations
- [ ] **Timeout handling** for long-running operations

### Security Testing

- [ ] **Input sanitization** for user prompts
- [ ] **Tool access control** 
- [ ] **Authentication flows** if applicable
- [ ] **Rate limiting** compliance

## Related Documentation

- [`📋 Pydantic Evals`](pydantic_ai_evals.md) - Model evaluation and benchmarking
- [`🔍 Observability`](pydantic_ai_observability.md) - Testing with monitoring tools
- [`🏗️ Implementation Plan`](pydantic_ai_implementation_plan.md) - Development workflow
- [`💻 API Documentation`](pydantic_ai_api.md) - Agent and model APIs