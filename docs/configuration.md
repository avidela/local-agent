# Configuration

The Local Agent's behavior can be configured using environment variables, typically set in a `.env` file in the root directory of the repository.

Here are the key environment variables and their descriptions:

*   `GOOGLE_GENAI_USE_VERTEXAI`: Set to `TRUE` to use Google Cloud Vertex AI for generative models, or `FALSE` (default) to use Google AI Gemini models directly.

*   `GOOGLE_API_KEY`: Your Google API Key. This is required for accessing Google AI models when `GOOGLE_GENAI_USE_VERTEXAI` is set to `FALSE`.

*   `VERTEXAI_PROJECT_ID`: Your Google Cloud Project ID. Required when `GOOGLE_GENAI_USE_VERTEXAI` is set to `TRUE`.

*   `VERTEXAI_LOCATION`: The Google Cloud location for Vertex AI. Required when `GOOGLE_GENAI_USE_VERTEXAI` is set to `TRUE`.

*   `REPO_ROOT`: The absolute path to the root directory the agent should have access to for file system operations.
    *   The path can use `~` for the user's home directory (it will be expanded).
    *   If not set, the Langchain file system tools (`ListDirectoryTool`, `ReadFileTool`, etc.) default to the current working directory where the agent was started.
    *   However, the custom tool wrappers (`grep_file`, `find_files`, `diff_files`) default to `/repos` if `REPO_ROOT` is not set.
    *   In the recommended Docker setup, this is typically handled by Docker volume mapping (mounting a host directory to `/repos` in the container), and `REPO_ROOT` might not need to be explicitly set in the `.env` file unless you want to override the default `/repos` path *within* the container.
    *   For local development, set this to the directory containing your projects (e.g., `/home/user/projects` or `~/Solutions`).

*   `AGENT_DIR`: Specifies the directory containing the agent definitions that `get_fast_api_app` should load. Defaults to `agents/adk` if not set.

*   `SESSION_DB_URL`: The database connection string for storing session history. If not set or empty, sessions will be stored in memory only (using `InMemorySessionService`). Example format: `postgresql://user:password@host:port/database`.

*   `SESSION_TYPE`: (Potentially used by ADK, check ADK docs) Specifies the type of session storage to use (e.g., `memory`, `database`). Often inferred from `SESSION_DB_URL`.

*   `ARTIFACT_TYPE`: (Potentially used by ADK, check ADK docs) Specifies the type of artifact storage to use (e.g., `memory`, `gcs`).

*   `GCS_BUCKET_NAME`: (Potentially used by ADK, check ADK docs) The name of the Google Cloud Storage bucket to use if `ARTIFACT_TYPE` is set to `gcs`.

*   `AGENT_MODEL`: (Optional) Specifies the model name (e.g., `gemini-2.5-pro-exp-03-25`) to use for the main `local_agent`. Overrides the default specified in `agent.py`.

*   `RESEARCHER_MODEL`: (Optional) Specifies the model name to use for the `researcher` sub-agent. Overrides the default specified in its `agent.py`.

*   `DEVELOPER_MODEL`: (Optional) Specifies the model name to use for the `developer` sub-agent. Overrides the default specified in its `agent.py`.

This list may not be exhaustive, and additional environment variables might be introduced for new features or tools. Refer to the specific tool or agent documentation for any tool-specific configuration options, and consult the Google Agent Development Kit (ADK) documentation for framework-level variables.
