# 🤖 Model Service & Provider Integration

> **Vertex AI and Anthropic integration using official Pydantic AI providers**

## 🎯 Overview

The Model Service provides a unified interface for accessing different AI model providers (Google Vertex AI and Anthropic) using official Pydantic AI provider patterns with proper authentication and configuration.

## 🚀 Model Service Implementation

### Core Model Service
```python
# src/services/model_service.py
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.instrumented import InstrumentedModel
from typing import Optional, Union, Dict, Any
import os

class ModelService:
    """
    Service for managing AI model providers using official Pydantic AI patterns.
    Supports Google Vertex AI (Gemini) and Anthropic (Claude) models.
    """
    
    def __init__(self):
        # Initialize providers with official API patterns
        self.google_provider = GoogleProvider(vertexai=True)  # Use Vertex AI
        self.anthropic_provider = AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Model configuration cache
        self._model_cache: Dict[str, Union[GoogleModel, AnthropicModel]] = {}
    
    def get_model(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        instrumented: bool = True
    ) -> Union[GoogleModel, AnthropicModel, InstrumentedModel]:
        """
        Get model instance with official Pydantic AI providers.
        
        Args:
            provider: 'google' or 'anthropic'
            model_name: Model identifier (e.g., 'gemini-1.5-flash', 'claude-3-sonnet')
            temperature: Model temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            instrumented: Whether to wrap with observability instrumentation
        
        Returns:
            Configured model instance
        """
        
        cache_key = f"{provider}_{model_name}_{temperature}_{max_tokens}"
        
        if cache_key not in self._model_cache:
            if provider == "google":
                base_model = self._create_google_model(
                    model_name, temperature, max_tokens
                )
            elif provider == "anthropic":
                base_model = self._create_anthropic_model(
                    model_name, temperature, max_tokens
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Supported providers: google, anthropic"
                )
            
            self._model_cache[cache_key] = base_model
        
        base_model = self._model_cache[cache_key]
        
        # Wrap with instrumentation for observability if requested
        if instrumented:
            from ..observability.instrumentation import observability
            instrumentation_settings = observability.get_custom_instrumentation_settings(
                include_binary_content=False
            )
            return InstrumentedModel(base_model, instrumentation_settings)
        
        return base_model
    
    def _create_google_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> GoogleModel:
        """Create Google Vertex AI model using official provider"""
        
        return GoogleModel(
            model_name,
            provider=self.google_provider,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    
    def _create_anthropic_model(
        self,
        model_name: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> AnthropicModel:
        """Create Anthropic model using official provider"""
        
        return AnthropicModel(
            model_name,
            provider=self.anthropic_provider,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def get_available_models(self) -> Dict[str, list]:
        """Get list of available models by provider"""
        
        return {
            "google": [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro"
            ],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        }
    
    def validate_model_config(self, provider: str, model_name: str) -> bool:
        """Validate if model configuration is supported"""
        
        available = self.get_available_models()
        return provider in available and model_name in available[provider]
```

## 🔧 Google Cloud Vertex AI Setup

### Authentication Configuration
```python
# src/services/google_auth.py
import os
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

class GoogleCloudAuthService:
    """Handle Google Cloud authentication for Vertex AI"""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.credentials = None
        
    def setup_credentials(self):
        """Setup Google Cloud credentials using Application Default Credentials"""
        
        try:
            # Use Application Default Credentials (ADC)
            # This works with:
            # - Service account key files (GOOGLE_APPLICATION_CREDENTIALS)
            # - gcloud auth application-default login
            # - Metadata service in GCP environments
            self.credentials, project = default()
            
            if not self.project_id:
                self.project_id = project
                
            return True
            
        except DefaultCredentialsError as e:
            raise ValueError(
                f"Google Cloud credentials not found. "
                f"Please set up authentication: {str(e)}"
            )
    
    def get_project_id(self) -> str:
        """Get configured Google Cloud project ID"""
        
        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable not set"
            )
        
        return self.project_id

# Global auth service instance
google_auth = GoogleCloudAuthService()
```

### Vertex AI Configuration
```python
# src/config/vertex_ai.py
import os
from typing import Dict, Any

class VertexAIConfig:
    """Configuration for Google Vertex AI integration"""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        
        configs = {
            "gemini-1.5-flash": {
                "max_output_tokens": 8192,
                "temperature_range": (0.0, 2.0),
                "supports_streaming": True,
                "supports_tools": True
            },
            "gemini-1.5-pro": {
                "max_output_tokens": 8192,
                "temperature_range": (0.0, 2.0),
                "supports_streaming": True,
                "supports_tools": True
            },
            "gemini-1.0-pro": {
                "max_output_tokens": 2048,
                "temperature_range": (0.0, 1.0),
                "supports_streaming": True,
                "supports_tools": False
            }
        }
        
        return configs.get(model_name, {})
    
    def validate_project_setup(self) -> bool:
        """Validate that Vertex AI is properly configured"""
        
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set")
        
        # Additional validation could include:
        # - API enablement check
        # - Service account permissions
        # - Quota validation
        
        return True

vertex_config = VertexAIConfig()
```

## 🔑 Anthropic API Setup

### Anthropic Configuration
```python
# src/config/anthropic.py
import os
from typing import Dict, Any

class AnthropicConfig:
    """Configuration for Anthropic API integration"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set"
            )
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        
        configs = {
            "claude-3-opus-20240229": {
                "max_tokens": 4096,
                "temperature_range": (0.0, 1.0),
                "supports_streaming": True,
                "supports_tools": True,
                "context_window": 200000
            },
            "claude-3-sonnet-20240229": {
                "max_tokens": 4096,
                "temperature_range": (0.0, 1.0),
                "supports_streaming": True,
                "supports_tools": True,
                "context_window": 200000
            },
            "claude-3-haiku-20240307": {
                "max_tokens": 4096,
                "temperature_range": (0.0, 1.0),
                "supports_streaming": True,
                "supports_tools": True,
                "context_window": 200000
            }
        }
        
        return configs.get(model_name, {})
    
    def validate_api_key(self) -> bool:
        """Validate API key format"""
        
        if not self.api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Anthropic API key format")
        
        return True

anthropic_config = AnthropicConfig()
```

## 🧪 Model Testing & Validation

### Model Test Service
```python
# src/services/model_test.py
from typing import Dict, Any, List
import asyncio
from .model_service import ModelService

class ModelTestService:
    """Service for testing and validating model configurations"""
    
    def __init__(self):
        self.model_service = ModelService()
    
    async def test_model_connection(
        self,
        provider: str,
        model_name: str
    ) -> Dict[str, Any]:
        """Test basic model connectivity"""
        
        try:
            model = self.model_service.get_model(
                provider=provider,
                model_name=model_name,
                instrumented=False  # Skip instrumentation for testing
            )
            
            # Simple test prompt
            from pydantic_ai import Agent
            test_agent = Agent(model)
            result = await test_agent.run("Hello, can you respond with 'OK'?")
            
            return {
                "success": True,
                "provider": provider,
                "model": model_name,
                "response": result.data,
                "cost": result.cost(),
                "message_count": len(result.all_messages())
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model_name,
                "error": str(e)
            }
    
    async def test_all_models(self) -> List[Dict[str, Any]]:
        """Test all available models"""
        
        available_models = self.model_service.get_available_models()
        results = []
        
        for provider, models in available_models.items():
            for model_name in models:
                result = await self.test_model_connection(provider, model_name)
                results.append(result)
        
        return results
    
    async def benchmark_model_performance(
        self,
        provider: str,
        model_name: str,
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """Benchmark model performance with test prompts"""
        
        model = self.model_service.get_model(
            provider=provider,
            model_name=model_name,
            instrumented=False
        )
        
        from pydantic_ai import Agent
        agent = Agent(model)
        
        results = []
        total_cost = 0.0
        
        for prompt in test_prompts:
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await agent.run(prompt)
                end_time = asyncio.get_event_loop().time()
                
                results.append({
                    "prompt": prompt,
                    "success": True,
                    "response_time": end_time - start_time,
                    "cost": result.cost(),
                    "message_count": len(result.all_messages()),
                    "response_length": len(str(result.data))
                })
                
                total_cost += result.cost()
                
            except Exception as e:
                end_time = asyncio.get_event_loop().time()
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                })
        
        return {
            "provider": provider,
            "model": model_name,
            "total_prompts": len(test_prompts),
            "successful_prompts": sum(1 for r in results if r["success"]),
            "total_cost": total_cost,
            "average_response_time": sum(r["response_time"] for r in results) / len(results),
            "results": results
        }

model_test_service = ModelTestService()
```

## 🌍 Environment Configuration

### Development Environment
```bash
# .env.development
# Google Cloud (Vertex AI)
GOOGLE_CLOUD_PROJECT=your-dev-project-id
VERTEX_AI_LOCATION=us-central1
# Use: gcloud auth application-default login

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# Optional: Service Account Key (alternative to ADC)
# GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
```

### Production Environment
```bash
# .env.production
# Google Cloud (Vertex AI)
GOOGLE_CLOUD_PROJECT=your-prod-project-id
VERTEX_AI_LOCATION=us-central1
# Use: Service Account with Workload Identity or service account key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-production-api-key

# Service Account Key for production
GOOGLE_APPLICATION_CREDENTIALS=/etc/gcp/service-account-key.json
```

## 🔧 Model Configuration Schema

### Model Configuration Types
```python
# src/schemas/models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any

class ModelConfig(BaseModel):
    """Model configuration schema"""
    
    provider: Literal["google", "anthropic"]
    model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "google",
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_tokens": 4096
            }
        }

class ModelResponse(BaseModel):
    """Model response schema"""
    
    provider: str
    model: str
    available: bool
    config: Dict[str, Any]
    error: Optional[str] = None

class ModelTestResult(BaseModel):
    """Model test result schema"""
    
    success: bool
    provider: str
    model: str
    response: Optional[str] = None
    cost: Optional[float] = None
    message_count: Optional[int] = None
    error: Optional[str] = None
    response_time: Optional[float] = None
```

## 🚀 Key Features

**✅ Official Provider Integration:**
- [`GoogleProvider(vertexai=True)`](https://docs.pydantic.ai/models/google/) for Vertex AI access
- [`AnthropicProvider`](https://docs.pydantic.ai/models/anthropic/) for direct Anthropic API
- [`InstrumentedModel`](https://docs.pydantic.ai/models/instrumented/) wrapper for observability

**🔐 Authentication & Security:**
- Google Cloud Application Default Credentials (ADC)
- Service account key support for production
- Secure API key management for Anthropic

**⚡ Performance & Reliability:**
- Model instance caching for efficiency
- Configuration validation and testing
- Comprehensive error handling

**📊 Monitoring & Testing:**
- Built-in model connectivity testing
- Performance benchmarking capabilities
- Cost and usage tracking

**🔧 Configuration Management:**
- Environment-specific configurations
- Model capability validation
- Provider-specific optimizations

---

*See [main index](./pydantic_ai_index.md) for complete implementation guide.*