# ADK Backend

This directory contains the ADK (Agent Development Kit) backend implementation for the Local Agent project.

## Overview

The ADK backend is built using Google's Agent Development Kit and provides the core agent functionality for the Local Agent project. It includes:

- Multi-agent architecture with specialized roles
- Tools for file system operations, search, and more
- API endpoints for interacting with the agent

## Directory Structure

```
adk/
├── Dockerfile            # Docker configuration for the ADK backend
├── api.py                # FastAPI application and API endpoints
├── pyproject.toml        # ADK-specific dependencies
├── .dockerignore         # Files to exclude from Docker builds
├── agents/               # Agent implementations
│   ├── multi_tool_agent/ # Main agent implementation
│   │   ├── __init__.py
│   │   ├── __main__.py   # Entry point for running the agent
│   │   ├── agent.py      # Defines the main agent and its components
│   │   ├── prompt.py     # Contains the instruction prompt for the agent
│   │   ├── callbacks/    # Callback handlers for agent lifecycle hooks
│   │   ├── sub_agents/   # Specialized sub-agents
│   │   └── tools/        # Tool implementations
│   └── oauth_calendar_agent/ # OAuth-enabled calendar agent
└── ...
```

## Running the ADK Backend

From the ADK directory:

```bash
uv run uvicorn api:app --reload --port 8001
```

From the root directory:

```bash
uv run uvicorn apps.backends.adk.api:app --reload --port 8001
```

## Dependencies

Dependencies for the ADK backend are managed in the `pyproject.toml` file in this directory. To update dependencies:

1. Edit the `pyproject.toml` file
2. Update the lock file:
   ```bash
   uv lock
   ```
3. Update your environment:
   ```bash
   # To sync only this workspace
   uv sync
   
   # Or from the root directory, to sync all workspaces at once
   cd ../../..
   uv sync --all-packages