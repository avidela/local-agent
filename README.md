# Local Agent

A multi-agent system built with Google's Agent Development Kit (ADK), featuring specialized sub-agents for research and development tasks.

## Quick Start with Docker

1. Create a `.env` file in the `/repos/local-agent` directory with the following content:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```
   Replace `YOUR_API_KEY_HERE` with your actual Google API key.

2. Build and run with Docker Compose:
   ```bash
   docker compose build
   docker compose up
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
   uv sync
   
   # Activate the virtual environment
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\\Scripts\\activate
   ```
   
   For development purposes, you can also install dev dependencies:
   ```bash
   uv sync --dev
   ```
   
4. Configure environment variables in `.env`
5. Run the application:
   ```bash
   uv run uvicorn api:app --reload --port 8001
   ```
   
## Updating Dependencies

When you need to add or update dependencies:

1. Edit `pyproject.toml` to add or modify dependencies
2. Update the lock file:
   ```bash
   uv lock
   ```
3. Update your environment:
   ```bash
   uv sync
   ```
4. Commit both pyproject.toml and uv.lock to version control

## Documentation

See the `docs/` directory for detailed documentation:
- [Project Overview](docs/overview.md)

## License

[Add appropriate license information here]