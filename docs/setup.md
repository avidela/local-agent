# Setup and Installation

This document provides instructions on how to set up and run the Local Agent project locally using Docker or a Python virtual environment.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Git
*   Docker and Docker Compose (for the Docker setup)
*   Python 3.12+ (for the local development setup)
*   UV (optional, but recommended for dependency management in local development)

## Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone <repository_url>
cd local-agent
```

Replace `<repository_url>` with the actual URL of the repository.

## Configuration (.env file)

Create a `.env` file in the root directory of the cloned repository (`/repos/local-agent` if you are following the Docker setup instructions, or the directory you cloned into for local development). This file will contain necessary environment variables.

```
# Set to FALSE to use Google AI Gemini models directly
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Your Google API Key (required for accessing Google AI models)
GOOGLE_API_KEY=YOUR_API_KEY_HERE

# The root directory the agent will have access to. 
# In the Docker setup, this is mapped to /repos.
# For local development, set this to the directory containing your projects.
# REPO_ROOT=/path/to/your/projects
```

Replace `YOUR_API_KEY_HERE` with your actual Google API key. If you are using the Docker setup, you typically don't need to set `REPO_ROOT` in the `.env` file as it's handled by the Docker Compose volume mapping.

## Setup with Docker Compose (Recommended)

This is the recommended way to run the Local Agent as it encapsulates the environment and dependencies.

1.  Make sure you have created the `.env` file as described above.
2.  Build and run the Docker containers using Docker Compose:

    ```bash
    docker compose build
    docker compose up
    ```

    This will build the agent image and start the necessary services (including a database if configured in `docker-compose-db.yml`). The local directory containing the `local-agent` repository will be mounted as `/app` inside the container, and the directory specified by the `volumes` mapping in `docker-compose.yml` (typically `~/Solutions`) will be mounted as `/repos`.

3.  The agent API should be accessible, typically at `http://localhost:8001` (check the `docker-compose.yml` for the exposed port).

## Local Development Setup (using a Virtual Environment)

If you prefer to run the agent directly on your system using a Python virtual environment:

1.  Make sure you have Python 3.12+ and UV installed.
2.  Navigate to the root directory of the cloned repository.
3.  Create a virtual environment and install dependencies using UV:

    ```bash
    # Create a virtual environment in the .venv directory and sync dependencies from pyproject.toml and uv.lock
    uv venv
    uv sync
    
    # Activate the virtual environment
    # On Linux/Mac:
    source .venv/bin/activate
    # On Windows:
    .venv\\Scripts\\activate
    ```

    If you plan on contributing to the agent's development, install the development dependencies as well:

    ```bash
    uv sync --dev
    ```

4.  Make sure you have configured the `.env` file as described above. For local development, you might need to explicitly set the `REPO_ROOT` environment variable in your `.env` file to the directory containing the projects you want the agent to access.

5.  Run the FastAPI application using uvicorn (from the root of the repository):

    ```bash
    uv run uvicorn apps.backends.adk.api:app --reload --port 8001
    ```

    The `--reload` flag is useful during development as it will restart the server whenever code changes are detected.

## Working with UV Workspaces

This project uses UV workspaces to isolate dependencies for different components:

1. **Root workspace**: Contains shared dependencies and workspace configuration
2. **Streamlit workspace**: Contains frontend-specific dependencies in `apps/frontends/streamlit/`
3. **ADK workspace**: Contains backend-specific dependencies in `apps/backends/adk/`

### Running Commands in Specific Workspaces

To run commands in a specific workspace:

```bash
# For the ADK backend
cd apps/backends/adk
uv run uvicorn api:app --reload --port 8001

# For the Streamlit frontend
cd apps/frontends/streamlit
uv run streamlit run main.py
```

### Updating Dependencies in Workspaces

When updating dependencies in a workspace:

1. Navigate to the workspace directory (e.g., `apps/backends/adk/` or `apps/frontends/streamlit/`)
2. Edit the workspace's `pyproject.toml` file
3. Update the workspace's lock file:

   ```bash
   uv lock
   ```

4. Update the workspace's dependencies:

   ```bash
   uv sync
   ```

5. Commit both the workspace's `pyproject.toml` and `uv.lock` to version control

### Root Workspace Dependencies

For dependencies that are shared across all workspaces:

1. Edit the root `pyproject.toml` file
2. Update the root lock file:

   ```bash
   uv lock
   ```

3. Update your virtual environment with the changes:

   ```bash
   # To sync only the root workspace
   uv sync
   
   # To sync all workspaces at once
   uv sync --all-packages
   ```

4. Commit both the root `pyproject.toml` and `uv.lock` to version control
