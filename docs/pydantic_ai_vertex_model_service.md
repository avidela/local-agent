# Vertex AI Model Service Implementation

## Model Service for Vertex AI

### Complete Model Service Implementation
```python
# src/services/model_service.py
from typing import Dict, Any, Optional
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..core.config import settings

class ModelService:
    """Service for managing Vertex AI model providers using Pydantic AI"""
    
    def __init__(self):
        self.project = settings.google_cloud_project
        self.region = settings.google_cloud_region
        
        # Initialize Google Vertex AI provider
        self.google_provider = GoogleProvider(
            vertexai=True,
            region=self.region,
            project_id=self.project
        )
        
        # Initialize Anthropic Vertex provider (if supported)
        # Note: Check Pydantic AI docs for Anthropic Vertex support
        self.anthropic_provider = AnthropicProvider()
        
        # Cache for initialized models
        self._model_cache: Dict[str, Any] = {}
    
    def get_model(
        self,
        provider: str,
        model_name: str,
        **config
    ):
        """Get a Pydantic AI model instance"""
        
        cache_key = f"{provider}:{model_name}:{hash(str(config))}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if provider == "vertex-gemini":
            model = self._create_gemini_model(model_name, **config)
        elif provider == "vertex-anthropic":
            model = self._create_anthropic_model(model_name, **config)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
        
        self._model_cache[cache_key] = model
        return model
    
    def _create_gemini_model(self, model_name: str, **config):
        """Create Vertex AI Gemini model using Pydantic AI"""
        return GoogleModel(
            model_name=model_name,
            provider=self.google_provider,
            **config
        )
    
    def _create_anthropic_model(self, model_name: str, **config):
        """Create Anthropic model (check if Vertex is supported)"""
        # Note: This may need adjustment based on Pydantic AI's Anthropic Vertex support
        return AnthropicModel(
            model_name=model_name,
            provider=self.anthropic_provider,
            **config
        )
    
    def list_available_models(self) -> Dict[str, list]:
        """List available models by provider"""
        return {
            "vertex-gemini": [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro"
            ],
            "vertex-anthropic": [
                "claude-3-5-sonnet@20241022",
                "claude-3-5-haiku@20241022",
                "claude-3-opus@20240229"
            ]
        }
    
    def validate_model(self, provider: str, model_name: str) -> bool:
        """Validate if model exists and is accessible"""
        available_models = self.list_available_models()
        
        if provider not in available_models:
            return False
            
        return model_name in available_models[provider]
```

### Authentication Setup
```python
# src/core/auth_setup.py
import os
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

def setup_vertex_auth():
    """Setup Vertex AI authentication using default credentials"""
    
    try:
        # Try to get default credentials
        credentials, project = default()
        
        if not project:
            raise ValueError("No project ID found in default credentials")
        
        return credentials, project
        
    except DefaultCredentialsError:
        raise ValueError(
            "No valid Google Cloud credentials found. "
            "Please ensure you have:\n"
            "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or\n"
            "2. Run 'gcloud auth application-default login', or\n"
            "3. Use a service account key file"
        )

def verify_vertex_access(project: str, region: str = "us-central1"):
    """Verify access to Vertex AI services"""
    try:
        from google.cloud import aiplatform
        
        aiplatform.init(project=project, location=region)
        
        # Test by listing models (this will fail if no access)
        client = aiplatform.gapic.ModelServiceClient()
        parent = f"projects/{project}/locations/{region}"
        
        # Just test the connection, don't need to iterate
        request = client.list_models(parent=parent, page_size=1)
        
        return True
        
    except Exception as e:
        raise ValueError(f"Cannot access Vertex AI: {str(e)}")
```

### Updated pyproject.toml Dependencies
```toml
# Additional dependencies for Vertex AI
dependencies = [
    # ... existing dependencies ...
    "google-cloud-aiplatform>=1.38.0",
    "anthropic[vertex]>=0.7.0",
    "google-auth>=2.23.0",
    "google-auth-oauthlib>=1.1.0",
    "google-auth-httplib2>=0.2.0",
    "google-cloud-core>=2.3.0",
    "protobuf>=4.21.0",
]
```

### Environment Setup Documentation
```bash
# Required environment variables for Vertex AI

# Option 1: Use service account key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Option 2: Use gcloud CLI authentication (for development)
gcloud auth application-default login

# Required project configuration
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_REGION="us-central1"  # or your preferred region
```

### Service Account Permissions
The service account needs these IAM roles:
- `roles/aiplatform.user` - For Vertex AI access
- `roles/ml.developer` - For model serving
- `roles/serviceusage.serviceUsageConsumer` - For API usage

### Integration with Agent Factory
```python
# Updated src/agents/dynamic/factory.py
class AgentFactory:
    def __init__(self, tool_registry: ToolRegistry, model_service: ModelService):
        self.tool_registry = tool_registry
        self.model_service = model_service
    
    async def create_agent(self, definition: AgentDefinition, user_id: UUID, db: AsyncSession) -> Agent:
        """Create a Pydantic AI agent from definition"""
        
        # Validate model availability
        if not self.model_service.validate_model(
            definition.model["provider"], 
            definition.model["model"]
        ):
            raise ValueError(f"Model not available: {definition.model['provider']}:{definition.model['model']}")
        
        # Get model using Vertex AI
        model = self.model_service.get_model(
            provider=definition.model["provider"],
            model_name=definition.model["model"],
            temperature=definition.model.get("temperature", 0.1),
            max_tokens=definition.model.get("max_tokens"),
        )
        
        # Create tools
        tools = []
        for tool_def in definition.tools:
            if tool_def["enabled"]:
                tool = self.tool_registry.get_tool(tool_def["name"])
                configured_tool = tool.configure(tool_def.get("config", {}))
                tools.append(configured_tool)
        
        # Create agent with Vertex AI model
        agent = Agent(
            model=model,
            system_prompt=definition.system_prompt,
            tools=tools,
            deps_type=type(None)
        )
        
        return agent