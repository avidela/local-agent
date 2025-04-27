# Project Overview

The Local Agent is a multi-agent system built using Google's Agent Development Kit (ADK). It is designed to assist with software development tasks by delegating to specialized sub-agents, such as a researcher and a developer.

The system can be run locally using Docker Compose, which sets up a container for the local agent and maps the local `~/Solutions` directory to `/repos` within the container. This allows the agent to access and manipulate files in the specified repository directory.

Key features:
- Multi-agent architecture with specialized roles.
- Built with Google's Agent Development Kit (ADK).
- Dockerized setup for easy deployment.
- Access to file system operations within the configured repository directory.

The primary entry point for the application is `main.py`, and the API is defined in `api.py`. Dependencies are managed using `uv` and defined in `pyproject.toml` with a lock file `uv.lock`.

## Directory Structure

```text
.
├── Dockerfile
├── README.md
├── api.py
├── docker-compose-adk.yml
├── docker-compose-db.yml
├── docker-compose.yml
├── entrypoint.sh
├── main.py
├── pyproject.toml
├── uv.lock
├── .dockerignore
├── .gitignore
├── .env
└── agents
    ├── adk
    │   └── multi_tool_agent
    │       ├── __init__.py
    │       ├── __main__.py         # Entry point for running the agent
    │       ├── agent.py            # Defines the main local_agent and its components
    │       ├── prompt.py           # Contains the detailed instruction prompt for the local_agent
    │       ├── sub_agents
    │       │   ├── __init__.py
    │       │   ├── developer
    │       │   │   ├── __init__.py
    │       │   │   ├── agent.py    # Defines the developer sub-agent
    │       │   │   └── prompt.py
    │       │   └── researcher
    │       │       ├── __init__.py
    │       │       ├── agent.py    # Defines the researcher sub-agent
    │       │       └── prompt.py
    │       └── tools
    │           ├── __init__.py
    │           ├── filesystem
    │           │   ├── __init__.py
    │           │   ├── diff_tool.py  # Wrapper for the system's diff command
    │           │   ├── find_tool.py  # Wrapper for the system's find command
    │           │   └── grep_tool.py  # Wrapper for the system's grep command
    │           ├── example
    │           │   ├── __init__.py
    │           │   ├── time.py
    │           │   └── weather.py
    │           └── search_agent_tool.py # Handles search-related tasks using Google Search
    ├── langchain
    └── pydanticai
```

## Detailed Documentation

*   [Setup and Installation](setup.md)
*   [Configuration](configuration.md)
*   [Agent Architecture](architecture.md)
*   [Adding New Tools](adding_tools.md)
*   [Adding New Sub-agents](adding_sub_agents.md)
