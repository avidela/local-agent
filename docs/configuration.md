# Configuration

The Local Agent's behavior can be configured using environment variables, typically set in a `.env` file in the root directory of the repository.

Here are the key environment variables and their descriptions:

*   `GOOGLE_GENAI_USE_VERTEXAI`: Set to `TRUE` to use Google Cloud Vertex AI for generative models, or `FALSE` (default) to use Google AI Gemini models directly.

*   `GOOGLE_API_KEY`: Your Google API Key. This is required for accessing Google AI models when `GOOGLE_GENAI_USE_VERTEXAI` is set to `FALSE`.

*   `VERTEXAI_PROJECT_ID`: Your Google Cloud Project ID. Required when `GOOGLE_GENAI_USE_VERTEXAI` is set to `TRUE`.

*   `VERTEXAI_LOCATION`: The Google Cloud location for Vertex AI. Required when `GOOGLE_GENAI_USE_VERTEXAI` is set to `TRUE`.

*   `REPO_ROOT`: The absolute path to the root directory the agent should have access to for file system operations. In the recommended Docker setup, this is typically handled by Docker volume mapping and might not need to be explicitly set in the `.env` file. For local development, set this to the directory containing your projects.

*   `DATABASE_URL`: (If using a database for session storage) The URL for connecting to the database. The format depends on the database type (e.g., `postgresql://user:password@host:port/database`).

*   `SESSION_TYPE`: (If using session storage) Specifies the type of session storage to use (e.g., `memory`, `database`).

*   `ARTIFACT_TYPE`: (If using artifact storage) Specifies the type of artifact storage to use (e.g., `memory`, `gcs`).

*   `GCS_BUCKET_NAME`: (If using GCS for artifact storage) The name of the Google Cloud Storage bucket to use for storing artifacts.

*   `AGENT_MODEL`: (Optional) Specifies the model to use for the main `local_agent`. If not set, a default model is used.

*   `RESEARCHER_MODEL`: (Optional) Specifies the model to use for the `researcher` sub-agent. If not set, a default model is used.

*   `DEVELOPER_MODEL`: (Optional) Specifies the model to use for the `developer` sub-agent. If not set, a default model is used.

This list may not be exhaustive, and additional environment variables might be introduced for new features or tools. Refer to the specific tool or agent documentation for any tool-specific configuration options.
