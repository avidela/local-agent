"""
Model provider services using official PydanticAI patterns.
"""

import os
from typing import Any, Dict, Optional, Union

from ...config import settings
from ...database.models import ModelProvider


class ModelProviderService:
    """Service for managing model providers with official PydanticAI integration."""
    
    def __init__(self):
        """Initialize model provider service and set up environment variables."""
        self._setup_environment()
        self._available_models = self._get_available_models()
    
    def _setup_environment(self) -> None:
        """Set up environment variables for model providers."""
        
        # Set OpenAI API key if configured
        if settings.models.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.models.openai_api_key.get_secret_value()
        
        # Set Anthropic API key if configured  
        if settings.models.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.models.anthropic_api_key.get_secret_value()
        
        # Set Google credentials if configured
        if settings.models.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.models.google_application_credentials
        
        if settings.models.google_project_id:
            os.environ["GOOGLE_PROJECT_ID"] = settings.models.google_project_id
        
        # Set Vertex AI configuration from environment (ADK pattern)
        google_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if google_project:
            os.environ["GOOGLE_PROJECT_ID"] = google_project
    
    def _get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models based on configured API keys."""
        models = {}
        
        # OpenAI models (if API key is configured)
        if settings.models.openai_api_key:
            openai_models = {
                "openai:gpt-4o": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "max_tokens": 4096,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 128000,
                },
                "openai:gpt-4o-mini": {
                    "provider": "openai", 
                    "name": "gpt-4o-mini",
                    "max_tokens": 16384,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 128000,
                },
                "openai:gpt-4-turbo": {
                    "provider": "openai",
                    "name": "gpt-4-turbo",
                    "max_tokens": 4096,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 128000,
                },
                "openai:gpt-3.5-turbo": {
                    "provider": "openai",
                    "name": "gpt-3.5-turbo",
                    "max_tokens": 4096,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": False,
                    "context_window": 16385,
                },
            }
            models.update(openai_models)
        
        # Anthropic models (if API key is configured)
        if settings.models.anthropic_api_key:
            anthropic_models = {
                "anthropic:claude-3-5-sonnet-20241022": {
                    "provider": "anthropic",
                    "name": "claude-3-5-sonnet-20241022",
                    "max_tokens": 8192,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 200000,
                },
                "anthropic:claude-3-5-haiku-20241022": {
                    "provider": "anthropic",
                    "name": "claude-3-5-haiku-20241022", 
                    "max_tokens": 8192,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 200000,
                },
                "anthropic:claude-3-opus-20240229": {
                    "provider": "anthropic",
                    "name": "claude-3-opus-20240229",
                    "max_tokens": 4096,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 200000,
                },
            }
            models.update(anthropic_models)
        
        # Google Vertex AI models (if credentials or Vertex AI is configured)
        google_configured = (
            settings.models.google_application_credentials or
            os.environ.get("GOOGLE_CLOUD_PROJECT") or
            settings.models.google_project_id
        )
        if google_configured:
            vertex_models = {
                "google:gemini-2.0-flash-exp": {
                    "provider": "google",
                    "name": "gemini-2.0-flash-exp",
                    "max_tokens": 8192,
                    "supports_tools": True,
                    "supports_streaming": True,
                    "supports_multimodal": True,
                    "context_window": 1000000,
                },
            }
            models.update(vertex_models)
        
        return models
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available for use."""
        return model_name in self._available_models
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with metadata."""
        return self._available_models.copy()
    
    def get_provider_for_model(self, model_name: str) -> Optional[ModelProvider]:
        """Get the provider enum for a model name."""
        if model_name.startswith("openai:"):
            return ModelProvider.OPENAI
        elif model_name.startswith("anthropic:"):
            return ModelProvider.ANTHROPIC
        elif model_name.startswith("google-vertex:"):
            return ModelProvider.GOOGLE
        return None
    
    def create_agent(
        self,
        model_name: str,
        system_prompt: str,
        tools: Optional[list] = None,
        output_type: Optional[Union[type, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 2,
        deps_type: Optional[type] = None,
        **kwargs
    ) -> Optional[Any]:  # Will be Agent once pydantic_ai is installed
        """
        Create PydanticAI Agent with specified configuration using official API.
        
        Args:
            model_name: Model name in format 'provider:model' (e.g., 'openai:gpt-4o')
            system_prompt: System prompt for the agent
            tools: List of tools to register with the agent
            output_type: Expected output type (Pydantic model or built-in type)
            temperature: Model temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            retries: Number of retries on failure
            deps_type: Type for dependency injection
            **kwargs: Additional arguments for Agent constructor
        
        Returns:
            Configured Agent instance or None if model not available
        """
        
        if not self.is_model_available(model_name):
            print(f"Model {model_name} is not available")
            return None
        
        try:
            # Extract just the model name for Vertex AI (no prefixes)
            actual_model_name = model_name
            if ":" in model_name:
                provider_prefix, model_part = model_name.split(":", 1)
                actual_model_name = model_part  # Use just the model name part
            
            # Build agent configuration
            agent_kwargs = {
                "model": actual_model_name,
                "system_prompt": system_prompt,
                "retries": retries,
            }
            
            # Add optional parameters
            if output_type:
                agent_kwargs["output_type"] = output_type
            
            if tools:
                agent_kwargs["tools"] = tools
            
            if deps_type:
                agent_kwargs["deps_type"] = deps_type
            
            # Add any additional kwargs
            agent_kwargs.update(kwargs)
            
            # Create agent using official PydanticAI API with proper Vertex AI provider
            try:
                from pydantic_ai import Agent
                
                # For Google models, use official Vertex AI provider pattern from docs
                if model_name.startswith("google:") or "gemini" in actual_model_name.lower():
                    from pydantic_ai.models.google import GoogleModel
                    from pydantic_ai.providers.google import GoogleProvider
                    
                    # Create Vertex AI provider as per official docs pattern
                    google_provider = GoogleProvider(vertexai=True)
                    
                    # Create GoogleModel with provider (temperature/tokens handled at run time)
                    google_model = GoogleModel(
                        actual_model_name,
                        provider=google_provider
                    )
                    
                    # Create agent with GoogleModel instance (remove model from kwargs)
                    agent_kwargs.pop("model", None)
                    agent = Agent(google_model, **agent_kwargs)
                else:
                    # For other providers, use string model name
                    agent = Agent(**agent_kwargs)
                    
            except ImportError as e:
                print(f"pydantic_ai not installed - agent creation will fail: {e}")
                return None
            except Exception as e:
                print(f"Failed to create agent with model {model_name}: {e}")
                return None
            
            # Note: temperature and max_tokens are passed during agent.run() calls
            # They don't need to be stored on the agent instance
            
            return agent
            
        except Exception as e:
            print(f"Failed to create agent with model {model_name}: {e}")
            return None
    
    def validate_model_config(self, provider: ModelProvider, model_name: str) -> bool:
        """Validate if model configuration is available."""
        full_model_name = f"{provider.value}:{model_name}"
        return self.is_model_available(full_model_name)
    
    def get_default_model(self) -> str:
        """Get the default model name from settings."""
        default = settings.models.default_model
        
        # Fallback to first available model if default is not available
        if not self.is_model_available(default):
            available = list(self._available_models.keys())
            return available[0] if available else "openai:gpt-4o"
        
        return default
    
    def get_model_limits(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific limits and capabilities."""
        return self._available_models.get(model_name, {
            "max_tokens": 4096,
            "supports_tools": False,
            "supports_streaming": True,
            "supports_multimodal": False,
            "context_window": 4096,
        })
    
    def get_supported_providers(self) -> list[ModelProvider]:
        """Get list of currently supported/configured providers."""
        providers = []
        
        if settings.models.openai_api_key:
            providers.append(ModelProvider.OPENAI)
        
        if settings.models.anthropic_api_key:
            providers.append(ModelProvider.ANTHROPIC)
        
        # Check for Google/Vertex AI configuration
        google_configured = (
            settings.models.google_application_credentials or
            os.environ.get("GOOGLE_CLOUD_PROJECT") or
            settings.models.google_project_id
        )
        if google_configured:
            providers.append(ModelProvider.GOOGLE)
        
        return providers
    
    def get_models_by_provider(self, provider: ModelProvider) -> Dict[str, Dict[str, Any]]:
        """Get all models for a specific provider."""
        prefix = f"{provider.value}:"
        return {
            name: info 
            for name, info in self._available_models.items() 
            if name.startswith(prefix)
        }


# Global model provider service instance
model_provider_service = ModelProviderService()