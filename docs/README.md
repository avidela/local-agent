# Local Agent Documentation

## Project Overview
This project is a multi-agent system built using Google's Agent Development Kit (ADK). It consists of a main agent and specialized sub-agents that handle different types of tasks:

1. **Main Agent** - Coordinates between sub-agents and handles direct queries
2. **Researcher Agent** - Handles information retrieval and research questions using Google Search
3. **Developer Agent** - Handles filesystem and development-related tasks

## Project Structure
```
/repos/local-agent/
├── .env                # Environment configuration
├── Dockerfile          # Docker container definition
├── docker-compose.yml  # Docker Compose configuration
├── pyproject.toml      # Project metadata and dependencies
├── api.py              # FastAPI application entry point
├── multi_tool_agent/
│   ├── __init__.py
│   ├── agent.py       # Main agent definition
│   ├── prompt.py      # System prompts
│   ├── tools/         # Tool implementations
│   └── sub_agents/    # Sub-agent implementations
│       ├── developer/
│       └── researcher/
```

## Current Capabilities
- Main agent coordinates between specialized sub-agents
- Researcher agent can perform Google searches
- Developer agent is set up for filesystem operations

## Current Model Configuration
- Main agent: gemini-2.5-flash-preview-04-17
- Sub-agents: gemini-2.5-pro-exp-03-25

## Deployment Options
- **Local Development**: Run directly with Python
- **Docker**: Run in a containerized environment (see [Docker Setup](docker-setup.md))