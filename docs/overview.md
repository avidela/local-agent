# Project Overview

The Local Agent is a multi-agent system built using Google's Agent Development Kit (ADK). It is designed to assist with software development tasks by delegating to specialized sub-agents, such as a researcher and a developer.

The system can be run locally using Docker Compose, which sets up a container for the local agent and maps the local `~/Solutions` directory to `/repos` within the container. This allows the agent to access and manipulate files in the specified repository directory.

Key features:
- Multi-agent architecture with specialized roles.
- Built with Google's Agent Development Kit (ADK).
- Dockerized setup for easy deployment.
- Access to file system operations within the configured repository directory.

The primary entry point for the application is `main.py`, and the API is defined in `api.py`. Dependencies are managed using `uv` and defined in `pyproject.toml` with a lock file `uv.lock`.