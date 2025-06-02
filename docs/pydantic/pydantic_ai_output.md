# PydanticAI Output Handling Guide

## Overview

This guide covers comprehensive output handling in PydanticAI, including structured outputs, streaming responses, output functions, and validation patterns. Understanding output handling is crucial for building robust AI applications that return the exact data formats your application needs.

## Core Output Concepts

### AgentRunResult and StreamedRunResult

All agent outputs are wrapped in result objects that provide access to:

- **[`result.output`](src/agents/agent.py)** - The actual output data
- **[`result.usage()`](src/utils/usage.py)** - Token usage and cost information  
- **[`result.all_messages()`](src/models/messages.py)** - Complete conversation history
- **Type preservation** - Generic typing maintains output type information

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('google-gla:gemini-1.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')

print(result.output)  # CityLocation(city='London', country='United Kingdom')
print(result.usage())  # Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

## Output Types

### Basic Output Types

#### Plain Text Output
When no [`output_type`](src/agents/agent.py) is specified or [`str`](src/types/output.py) is included:

```python
agent = Agent('openai:gpt-4o-mini')  # Defaults to str output
result = agent.run_sync('Tell me a joke')
print(result.output)  # Plain text string
```

#### Structured Data Output
Force structured responses using Pydantic models:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class BoxDimensions(BaseModel):
    width: int
    height: int
    depth: int
    units: str

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=BoxDimensions,
    system_prompt="Extract box dimensions from user input"
)

result = agent.run_sync('The box is 10x20x30 cm')
print(result.output)  # BoxDimensions(width=10, height=20, depth=30, units='cm')
```

### Union Output Types

#### Mixed Output Types
Allow either text or structured data:

```python
from typing import Union
from pydantic import BaseModel
from pydantic_ai import Agent

class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[Box, str],  # Can return either Box or str
    system_prompt=(
        "Extract box dimensions if possible, "
        "otherwise ask for clarification."
    ),
)

# Insufficient data - returns str
result = agent.run_sync('The box is 10x20x30')
print(result.output)  # "Please provide the units..."

# Complete data - returns Box
result = agent.run_sync('The box is 10x20x30 cm')  
print(result.output)  # Box(width=10, height=20, depth=30, units='cm')
```

#### Multiple Structured Types
Register multiple output tools for complex scenarios:

```python
from typing import Union
from pydantic_ai import Agent

# Type checking requires explicit generic parameters for unions
agent = Agent[None, Union[list[str], list[int]]](
    'openai:gpt-4o-mini',
    output_type=Union[list[str], list[int]],  # type: ignore
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.output)  # ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.output)  # [10, 20, 30]
```

## Output Functions

### Function-Based Output Processing

Output functions allow custom processing of model arguments and can hand off to other agents:

```python
import re
from typing import Union
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

class Row(BaseModel):
    name: str
    country: str

# Simulated database
tables = {
    'capital_cities': [
        Row(name='Amsterdam', country='Netherlands'),
        Row(name='Mexico City', country='Mexico'),
    ]
}

class SQLFailure(BaseModel):
    """Unrecoverable SQL failure."""
    explanation: str

def run_sql_query(query: str) -> list[Row]:
    """Execute SQL query on database."""
    select_table = re.match(r'SELECT (.+) FROM (\w+)', query)
    if select_table:
        column_names = select_table.group(1)
        if column_names != '*':
            raise ModelRetry("Only 'SELECT *' supported")
        
        table_name = select_table.group(2)
        if table_name not in tables:
            raise ModelRetry(f"Unknown table '{table_name}'")
        
        return tables[table_name]
    
    raise ModelRetry(f"Unsupported query: '{query}'")

sql_agent = Agent[None, Union[list[Row], SQLFailure]](
    'openai:gpt-4o',
    output_type=[run_sql_query, SQLFailure],
    instructions='SQL agent that runs queries on database.',
)
```

### Agent Hand-off with Output Functions

```python
from pydantic_ai._output import ToolRetryError
from pydantic_ai.exceptions import UnexpectedModelBehavior

async def hand_off_to_sql_agent(ctx: RunContext, query: str) -> list[Row]:
    """Convert natural language to SQL and execute."""
    # Drop final message to avoid passing tool call to SQL agent
    messages = ctx.messages[:-1]
    
    try:
        result = await sql_agent.run(query, message_history=messages)
        output = result.output
        
        if isinstance(output, SQLFailure):
            raise ModelRetry(f'SQL agent failed: {output.explanation}')
        
        return output
    except UnexpectedModelBehavior as e:
        # Bubble up retryable errors
        if (cause := e.__cause__) and isinstance(cause, ToolRetryError):
            raise ModelRetry(f'SQL agent failed: {cause.tool_retry.content}') from e
        else:
            raise

class RouterFailure(BaseModel):
    """Router agent failure message."""
    explanation: str

router_agent = Agent[None, Union[list[Row], RouterFailure]](
    'openai:gpt-4o',
    output_type=[hand_off_to_sql_agent, RouterFailure],
    instructions='Router to other agents. Never solve problems directly.',
)

result = router_agent.run_sync('Select all capital cities')
print(result.output)
# [Row(name='Amsterdam', country='Netherlands'), Row(name='Mexico City', country='Mexico')]
```

## Output Validation

### Asynchronous Output Validators

For validation requiring IO operations or complex business logic:

```python
from typing import Union
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from fake_database import DatabaseConn, QueryError

class Success(BaseModel):
    sql_query: str

class InvalidRequest(BaseModel):
    error_message: str

Output = Union[Success, InvalidRequest]

agent = Agent[DatabaseConn, Output](
    'google-gla:gemini-1.5-flash',
    output_type=Output,  # type: ignore
    deps_type=DatabaseConn,
    system_prompt='Generate PostgreSQL SQL queries based on user input.',
)

@agent.output_validator
async def validate_sql(ctx: RunContext[DatabaseConn], output: Output) -> Output:
    """Validate SQL queries by running EXPLAIN."""
    if isinstance(output, InvalidRequest):
        return output
    
    try:
        await ctx.deps.execute(f'EXPLAIN {output.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output

result = agent.run_sync(
    'get me users who were last active yesterday.', 
    deps=DatabaseConn()
)
print(result.output)
# Success(sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day')
```

## Streaming Outputs

### Text Streaming

#### Basic Text Streaming
Stream text responses as they're generated:

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')

async def stream_text_demo():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text():
            print(message)
            # Progressively prints:
            # "The first known"
            # "The first known use of "hello,"
            # "The first known use of "hello, world" was in"
            # ... (continues building up text)
```

#### Delta Text Streaming
Stream only text changes (deltas):

```python
async def stream_delta_demo():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text(delta=True):
            print(message)
            # Prints only new text:
            # "The first known"
            # " use of "hello,"
            # " world" was in"
            # ... (only differences)
```

> **Important**: When using [`delta=True`](src/streaming/delta.py), the final output message is NOT added to [`result.messages`](src/models/messages.py).

### Structured Data Streaming

#### TypedDict Streaming
Best for streaming structured data with partial validation:

```python
from datetime import date
from typing_extensions import TypedDict
from pydantic_ai import Agent

class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str

agent = Agent(
    'openai:gpt-4o',
    output_type=UserProfile,
    system_prompt='Extract user profile from input',
)

async def stream_profile_demo():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like dogs.'
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream():
            print(profile)
            # Progressively builds:
            # {'name': 'Ben'}
            # {'name': 'Ben', 'dob': date(1990, 1, 28)}
            # {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes dogs'}
```

#### Fine-Grained Validation Control
Handle validation errors during streaming:

```python
from pydantic import ValidationError
async def stream_with_validation():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like dogs.'
    async with agent.run_stream(user_input) as result:
        async for message, last in result.stream_structured(debounce_by=0.01):
            try:
                profile = await result.validate_structured_output(
                    message,
                    allow_partial=not last,  # Allow partial until final message
                )
                print(profile)
            except ValidationError:
                continue  # Skip invalid partial data
```

## Type Checking Considerations

### Generic Agent Parameters

For complex output types, explicit generic parameters help type checkers:

```python
from typing import Union
from pydantic_ai import Agent

# Type checker needs help with unions
agent = Agent[None, Union[list[str], list[int]]](
    'openai:gpt-4o-mini',
    output_type=Union[list[str], list[int]],  # type: ignore
    system_prompt='Extract colors or sizes',
)
```

### Common Type Checking Scenarios

1. **Union Types**: Use `# type: ignore` and explicit generic parameters
2. **List of Types** (mypy): Requires explicit generics  
3. **Async Output Functions** (mypy): Requires explicit generics
4. **Complex Function Types**: Use explicit generics for clarity

## Integration Patterns

### Database Integration

```python
async def save_structured_output():
    """Save agent output to database."""
    result = await agent.run('Extract user data: John, 25, Engineer')
    
    if isinstance(result.output, UserProfile):
        await db.users.create(result.output.dict())
        print(f"Saved user: {result.output.name}")
    else:
        print(f"Failed to extract: {result.output}")
```

### Multi-Agent Workflows

```python
async def processing_pipeline():
    """Chain multiple agents with different output types."""
    # Extract data
    extract_result = await extractor_agent.run(raw_input)
    
    # Validate data  
    validate_result = await validator_agent.run(
        f"Validate: {extract_result.output}",
        message_history=extract_result.all_messages()
    )
    
    # Transform data
    final_result = await transformer_agent.run(
        f"Transform: {validate_result.output}"
    )
    
    return final_result.output
```

### Error Handling Patterns

```python
from pydantic_ai.exceptions import UnexpectedModelBehavior

async def robust_output_handling():
    """Handle various output scenarios robustly."""
    try:
        result = await agent.run(user_input)
        
        if isinstance(result.output, SuccessType):
            return await process_success(result.output)
        elif isinstance(result.output, ErrorType):
            return await handle_error(result.output)
        else:
            return await handle_unexpected(result.output)
            
    except UnexpectedModelBehavior as e:
        logger.error(f"Model behavior error: {e}")
        return await fallback_processing(user_input)
```

## Best Practices

### Output Type Design

1. **Use Pydantic models** for complex structured data
2. **Provide clear field descriptions** to guide model responses
3. **Use unions sparingly** - prefer output functions for complex routing
4. **Include error types** in unions for graceful failure handling

### Validation Strategy

1. **Use output validators** for async/IO-heavy validation
2. **Implement retry logic** with [`ModelRetry`](src/exceptions/retry.py) for recoverable errors
3. **Validate early** - catch issues before expensive processing
4. **Log validation failures** for debugging and monitoring

### Streaming Best Practices

1. **Use TypedDict** for streamable structured data
2. **Handle validation errors** gracefully during streaming
3. **Implement debouncing** for high-frequency updates
4. **Consider delta streaming** for large text outputs

### Performance Optimization

1. **Minimize output type complexity** to reduce tool schema size
2. **Use output functions** instead of complex unions when possible
3. **Cache validation results** for repeated patterns
4. **Monitor token usage** with [`result.usage()`](src/utils/usage.py)

## Related Documentation

- [`🧪 Unit Testing Guide`](pydantic_ai_testing.md) - Testing output types and validation
- [`💬 Session & Messages`](pydantic_ai_sessions.md) - Managing conversation history
- [`🛠️ Agents & Tools`](pydantic_ai_agents.md) - Tool outputs vs agent outputs
- [`🔍 Observability`](pydantic_ai_observability.md) - Monitoring output patterns
- [`💻 API Documentation`](pydantic_ai_api.md) - HTTP API output handling