# Local Agent

A multi-agent system built with Google's Agent Development Kit (ADK), featuring specialized sub-agents for research and development tasks.

## Quick Start

1. **Both Docker users and local developers need to create a `.env` file**. Copy the `env/.env.template` file to `.env` in the root directory (`/repos/local-agent`) and replace `YOUR_API_KEY_HERE` with your actual Google API key.
   ```bash
   cp env/.env.template .env
   ```
   
   **Using Vertex AI (Optional):**
   If you prefer to use Vertex AI instead of a Google API Key, modify your `.env` file as follows:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=TRUE
   GOOGLE_API_KEY=
   GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID # UNCOMMENT AND REPLACE WITH YOUR PROJECT ID
   GOOGLE_CLOUD_LOCATION=LOCATION # UNCOMMENT AND REPLACE WITH YOUR PROJECT LOCATION (e.g., us-central1)
   ```
   Ensure you uncomment and replace `YOUR_PROJECT_ID` and `LOCATION` with your specific Google Cloud project details.

## Using Docker

1. Copy the template Docker Compose file to `docker-compose.yml`:

    ```bash
    cp docker-compose-template.yml docker-compose.yml
    ```

2. Customize the `docker-compose.yml` file to match your local environment.  **Important:** To allow the agent to access your local directories for file system operations, you need to update the volume mount in `docker-compose.yml`. Modify the `volumes` section under the `local-agent` service to point to the desired directory on your host machine. For example:

    ```yaml
    services:
      local-adk-agent:
        volumes:
          - /path/to/your/local/directory:/repos
    ```

    Replace `/path/to/your/local/directory` with the actual path to the directory you want to share with the agent.

3. Build and run with Docker Compose:

    ```bash
    docker compose build
    docker compose up -d
    ```

## Local Development

1. Set up a Python environment (3.12+ required as specified in pyproject.toml)
2. Install UV if you don't have it:
   ```bash
   curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
   ```
3. Create a virtual environment and install dependencies with UV:
   ```bash
   # Create a virtual environment and sync dependencies from lock file
   uv venv
   
   # Activate the virtual environment
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\\Scripts\\activate

   # install all dependencies
   uv sync
   ```
   
   For development purposes, you can also install dev dependencies:
   ```bash
   uv sync --dev
   ```
   
4. Configure environment variables in `.env`
5. Run the application:

   For the ADK backend:
   ```bash
   # From the root directory
   uv run uvicorn apps.backends.adk.api:app --reload --port 8001
   
   # Or navigate to the ADK directory and run
   cd apps/backends/adk
   uv run uvicorn api:app --reload --port 8001
   ```
   
   For the Streamlit frontend:
   ```bash
   # From the root directory
   uv run streamlit run apps/frontends/streamlit/main.py
   
   # Or navigate to the Streamlit directory and run
   cd apps/frontends/streamlit
   uv run streamlit run main.py
   ```
   
## Project Structure

This project uses UV workspaces to isolate dependencies for different components:

- **Root workspace**: Contains shared dependencies and workspace configuration
- **ADK backend**: Located in `apps/backends/adk/` with its own dependencies
- **Streamlit frontend**: Located in `apps/frontends/streamlit/` with its own dependencies

## Updating Dependencies

When you need to add or update dependencies:

### For the root workspace:

1. Edit the root `pyproject.toml` to add or modify shared dependencies
2. Update the lock file:
   ```bash
   uv lock
   ```
3. Update your environment:
   ```bash
   # To sync only the root workspace
   uv sync
   
   # To sync all workspaces at once
   uv sync --all-packages
   ```

### For a specific workspace (ADK or Streamlit):

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

Always commit both pyproject.toml and uv.lock files to version control.

## Documentation

See the `docs/` directory for detailed documentation:
- [Project Overview](docs/overview.md)

## License

[Add appropriate license information here]