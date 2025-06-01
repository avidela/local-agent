"""
Custom Pydantic validators for enhanced input validation.
"""

import re
from typing import Any, Dict, List
from pydantic import validator, Field

from ..database.models import ModelProvider


def validate_email(email: str) -> str:
    """Enhanced email validation."""
    email = str(email).strip()
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise ValueError('Invalid email format')
    
    # Check for common disposable email domains
    disposable_domains = [
        '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
        'mailinator.com', 'yopmail.com', 'throwaway.email'
    ]
    
    domain = email.split('@')[1].lower()
    if domain in disposable_domains:
        raise ValueError('Disposable email addresses are not allowed')
    
    return email.lower()


def validate_password_strength(password: str) -> str:
    """Enhanced password validation."""
    password = str(password).strip()
    
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters long')
    
    if len(password) > 128:
        raise ValueError('Password must not exceed 128 characters')
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        raise ValueError('Password must contain at least one uppercase letter')
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        raise ValueError('Password must contain at least one lowercase letter')
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        raise ValueError('Password must contain at least one digit')
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValueError('Password must contain at least one special character')
    
    # Check for common weak patterns
    weak_patterns = [
        r'(.)\1{3,}',  # Four or more consecutive identical characters
        r'123456',     # Sequential numbers
        r'abcdef',     # Sequential letters
        r'qwerty',     # Keyboard patterns
        r'password',   # Common words
        r'admin',
        r'user',
    ]
    
    password_lower = password.lower()
    for pattern in weak_patterns:
        if re.search(pattern, password_lower):
            raise ValueError('Password contains weak patterns and is not secure')
    
    return password


def validate_tool_name(tool_name: str) -> str:
    """Validate tool name format and security."""
    tool_name = str(tool_name).strip()
    
    if not tool_name:
        raise ValueError('Tool name cannot be empty')
    
    if len(tool_name) > 100:
        raise ValueError('Tool name must not exceed 100 characters')
    
    # Allow alphanumeric, underscore, hyphen, and dot
    if not re.match(r'^[a-zA-Z0-9_.-]+$', tool_name):
        raise ValueError('Tool name can only contain letters, numbers, underscores, hyphens, and dots')
    
    # Must start with a letter
    if not tool_name[0].isalpha():
        raise ValueError('Tool name must start with a letter')
    
    # Check for reserved names
    reserved_names = [
        'admin', 'root', 'system', 'config', 'settings',
        'api', 'auth', 'login', 'register', 'password',
        'user', 'users', 'agent', 'agents', 'session',
        'sessions', 'eval', 'evaluation', 'workflow'
    ]
    
    if tool_name.lower() in reserved_names:
        raise ValueError(f'Tool name "{tool_name}" is reserved and cannot be used')
    
    return tool_name


def validate_tool_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tool configuration for security and correctness."""
    if not isinstance(config, dict):
        raise ValueError('Tool configuration must be a dictionary')
    
    # Check for dangerous configuration keys
    dangerous_keys = [
        'exec', 'eval', 'system', 'subprocess', 'os',
        'file', 'open', 'read', 'write', 'delete',
        'password', 'secret', 'key', 'token', 'credential'
    ]
    
    for key in config.keys():
        if any(dangerous in key.lower() for dangerous in dangerous_keys):
            raise ValueError(f'Configuration key "{key}" is not allowed for security reasons')
    
    # Validate configuration values
    for key, value in config.items():
        if isinstance(value, str):
            # Check for potential code injection
            if any(pattern in value.lower() for pattern in ['__import__', 'exec(', 'eval(', 'os.system']):
                raise ValueError(f'Configuration value for "{key}" contains potentially dangerous content')
    
    return config


def validate_system_prompt(prompt: str) -> str:
    """Validate system prompt for safety and format."""
    prompt = str(prompt).strip()
    
    if not prompt:
        raise ValueError('System prompt cannot be empty')
    
    if len(prompt) < 10:
        raise ValueError('System prompt must be at least 10 characters long')
    
    if len(prompt) > 10000:
        raise ValueError('System prompt must not exceed 10,000 characters')
    
    # Check for potential prompt injection patterns
    dangerous_patterns = [
        r'ignore\s+previous\s+instructions',
        r'forget\s+everything',
        r'act\s+as\s+if',
        r'pretend\s+to\s+be',
        r'roleplay\s+as',
        r'system\s*:\s*override',
        r'admin\s+mode',
        r'developer\s+mode'
    ]
    
    prompt_lower = prompt.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt_lower):
            raise ValueError('System prompt contains potentially unsafe instructions')
    
    return prompt


def validate_model_name(model_name: str, provider: ModelProvider) -> str:
    """Validate model name format and availability."""
    model_name = str(model_name).strip()
    
    if not model_name:
        raise ValueError('Model name cannot be empty')
    
    if len(model_name) > 100:
        raise ValueError('Model name must not exceed 100 characters')
    
    # Provider-specific validation
    if provider == ModelProvider.GOOGLE:
        # Google models should follow specific patterns
        valid_patterns = [
            r'^gemini-.*',
            r'^text-.*',
            r'^chat-.*'
        ]
        if not any(re.match(pattern, model_name) for pattern in valid_patterns):
            raise ValueError(f'Invalid Google model name format: {model_name}')
    
    elif provider == ModelProvider.OPENAI:
        # OpenAI models should follow specific patterns
        valid_patterns = [
            r'^gpt-.*',
            r'^text-.*',
            r'^davinci.*',
            r'^curie.*',
            r'^babbage.*',
            r'^ada.*'
        ]
        if not any(re.match(pattern, model_name) for pattern in valid_patterns):
            raise ValueError(f'Invalid OpenAI model name format: {model_name}')
    
    elif provider == ModelProvider.ANTHROPIC:
        # Anthropic models should follow specific patterns
        valid_patterns = [
            r'^claude-.*',
            r'^haiku.*',
            r'^sonnet.*',
            r'^opus.*'
        ]
        if not any(re.match(pattern, model_name) for pattern in valid_patterns):
            raise ValueError(f'Invalid Anthropic model name format: {model_name}')
    
    return model_name


def validate_session_id(session_id: str) -> str:
    """Validate session ID format."""
    session_id = str(session_id).strip()
    
    if not session_id:
        raise ValueError('Session ID cannot be empty')
    
    # Should be a valid UUID format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, session_id.lower()):
        raise ValueError('Session ID must be a valid UUID')
    
    return session_id.lower()


def validate_workflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate workflow configuration."""
    if not isinstance(config, dict):
        raise ValueError('Workflow configuration must be a dictionary')
    
    required_keys = ['nodes', 'edges']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Workflow configuration must contain "{key}"')
    
    # Validate nodes
    nodes = config.get('nodes', [])
    if not isinstance(nodes, list):
        raise ValueError('Workflow nodes must be a list')
    
    if len(nodes) == 0:
        raise ValueError('Workflow must contain at least one node')
    
    if len(nodes) > 100:
        raise ValueError('Workflow cannot contain more than 100 nodes')
    
    # Validate edges
    edges = config.get('edges', [])
    if not isinstance(edges, list):
        raise ValueError('Workflow edges must be a list')
    
    return config


def validate_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate evaluation configuration."""
    if not isinstance(config, dict):
        raise ValueError('Evaluation configuration must be a dictionary')
    
    # Check for required evaluation parameters
    if 'type' not in config:
        raise ValueError('Evaluation configuration must specify a type')
    
    eval_type = config.get('type')
    if eval_type not in ['accuracy', 'latency', 'cost', 'custom']:
        raise ValueError('Evaluation type must be one of: accuracy, latency, cost, custom')
    
    return config