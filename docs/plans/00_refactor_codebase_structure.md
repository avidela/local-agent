# Task 00: Refactor Codebase Structure

This document outlines the planned refactoring of the project structure into a monorepo to better accommodate multiple backend implementations (ADK, LangGraph, Pydantic) and multiple frontend applications (Mesop, React, Streamlit), while promoting code reuse through shared packages.

## Proposed Monorepo Structure

```text
/your-project-root/
├── apps/                     # Deployable applications
│   ├── backends/             # Backend implementations
│   │   ├── adk/              # ADK-based backend
│   │   │   ├── Dockerfile
│   │   │   ├── api.py        # Uses ADK get_fast_api_app, imports from packages/*
│   │   │   ├── pyproject.toml # Dependencies: adk, fastapi, packages/agents/core, packages/agents/tools
│   │   │   └── .env
│   │   ├── langgraph/        # LangGraph/Langchain backend
│   │   │   ├── Dockerfile
│   │   │   ├── main.py       # FastAPI/LangServe app using LangGraph, imports from packages/*
│   │   │   ├── pyproject.toml # Dependencies: langchain, langgraph, fastapi, packages/agents/core, packages/agents/tools
│   │   │   └── .env
│   │   └── pydantic/         # Pure FastAPI + Pydantic backend (example)
│   │       ├── Dockerfile
│   │       ├── main.py       # FastAPI app, imports from packages/*
│   │       ├── pyproject.toml # Dependencies: fastapi, pydantic, packages/agents/core, packages/agents/tools
│   │       └── .env
│   └── frontends/            # Frontend implementations
│       ├── mesop/            # Mesop frontend
│       │   ├── main.py       # Mesop application code, calls a backend API
│       │   └── pyproject.toml # Dependencies: mesop, httpx (or similar)
│       ├── react/            # React frontend
│       │   ├── public/
│       │   ├── src/          # React components, hooks, API clients
│       │   ├── package.json  # Dependencies: react, axios
│       │   └── vite.config.js # Or other build tool config
│       └── streamlit/        # Streamlit frontend
│           ├── app.py        # Streamlit application code, calls a backend API
│           └── requirements.txt # Dependencies: streamlit, requests (or similar)
│
├── packages/                 # Shared libraries/code (not directly deployed)
│   ├── agents/               # Shared code specific to agent logic
│   │   ├── core/             # Core Python logic shared across backends
│   │   │   ├── __init__.py
│   │   │   ├── prompts/      # Directory for all agent prompts
│   │   │   ├── agents/       # Base agent definitions/logic (framework agnostic if possible)
│   │   │   └── pyproject.toml # Minimal dependencies, maybe pydantic for models
│   │   └── tools/            # Shared Python tool implementations
│   │       ├── __init__.py
│   │       ├── filesystem/   # e.g., grep_tool.py, find_tool.py, diff_tool.py
│   │       ├── search/       # e.g., search_tool.py
│   │       └── pyproject.toml # Dependencies needed for tools (e.g., subprocess)
│   └── api-schemas/          # Shared API request/response models (e.g., Pydantic models or OpenAPI specs)
│       ├── __init__.py       # Useful if backends aim for API consistency
│       └── pyproject.toml    # Dependency: pydantic
│
├── docs/                     # Documentation
│   ├── overview.md
│   ├── setup.md
│   ├── architecture.md
│   ├── configuration.md
│   ├── api.md
│   ├── tools.md
│   ├── callbacks.md
│   ├── plans/                # Directory for planned refactors/work
│   │   ├── 00_refactor_codebase_structure.md # This file
│   │   └── 00_refactor_agent_prompt.md       # Prompt for agent execution
│   └── ...                   # Other documentation files
│
├── .gitignore                # Git ignore rules
├── README.md                 # Root README explaining the monorepo structure, setup, etc.
├── pyproject.toml            # Root Python config (e.g., for shared dev tools like Ruff, Black)
└── package.json              # Root Node.js config (e.g., for managing workspaces, shared scripts)
```

## Key Principles

*   **Separation:** Clear separation between deployable `apps` and shared `packages`. Further separation within `apps` into `backends` and `frontends`.
*   **Code Reuse:** Core agent logic, prompts, tools, and API schemas are shared via `packages`. The `packages/agents/` directory specifically groups shared agent-related code.
*   **Independence:** Each app in `apps/` manages its own specific dependencies and deployment configuration (Dockerfile, etc.).
*   **Shared Data Models & Abstractable APIs:** Use shared data models (`packages/api-schemas`) for core message content. While backend frameworks may expose different endpoint structures, aim to make these differences abstractable by frontend clients, facilitating easier switching between backend implementations.

## Implementation Steps

*(Note: These steps are designed to be executed by an agent. A suggested prompt for instructing the agent can be found in [00_refactor_agent_prompt.md](00_refactor_agent_prompt.md).)*

This plan details moving the existing ADK-based agent (`local-agent`) into the new monorepo structure. It assumes the current code is in a directory named `local-agent` and the new structure is being created in a parent directory called `monorepo-root`.

**Phase 1: Create New Structure & Move Files**

1.  **Create Monorepo Root & Basic Structure:**
    *   Outside the current `local-agent` directory, create the main project root: `mkdir monorepo-root`
    *   `cd monorepo-root`
    *   Create top-level directories: `mkdir -p apps/backends/adk apps/frontends packages/agents/core packages/agents/tools packages/api-schemas docs/plans` (Ensure docs/plans exists)
    *   Create necessary subdirectories within packages:
        *   `mkdir -p packages/agents/core/prompts packages/agents/core/agents`
        *   `mkdir -p packages/agents/tools/filesystem packages/agents/tools/search`

2.  **Move Existing Documentation:**
    *   Move the entire contents of `../local-agent/docs/` (excluding the old `plans` dir if it exists) into `monorepo-root/docs/`.
    *   `mv ../local-agent/docs/* ./docs/`
    *   *(Verify the `plans/` directory and its contents were moved correctly into `monorepo-root/docs/plans/`)*

3.  **Move Core Agent Logic to `packages/agents/core`:**
    *   Move prompts: `mv ../local-agent/agents/adk/multi_tool_agent/prompt.py ./packages/agents/core/prompts/local_agent_prompt.py`
    *   Move sub-agent definitions: `mv ../local-agent/agents/adk/multi_tool_agent/sub_agents/* ./packages/agents/core/agents/` (Moves `developer/` and `researcher/`)

4.  **Move Tool Logic to `packages/agents/tools`:**
    *   Move filesystem tools: `mv ../local-agent/agents/adk/multi_tool_agent/tools/filesystem/* ./packages/agents/tools/filesystem/`
    *   Move search tool: `mv ../local-agent/agents/adk/multi_tool_agent/tools/search_agent_tool.py ./packages/agents/tools/search/`
    *   *(Note: Leaving `example/` tools behind for now)*

5.  **Move ADK Application Code to `apps/backends/adk`:**
    *   Move main agent definition: `mv ../local-agent/agents/adk/multi_tool_agent/agent.py ./apps/backends/adk/`
    *   Move API entry point: `mv ../local-agent/api.py ./apps/backends/adk/`
    *   Move main runner: `mv ../local-agent/main.py ./apps/backends/adk/`
    *   Move callbacks: `mv ../local-agent/agents/adk/multi_tool_agent/callbacks ./apps/backends/adk/`
    *   Move build/config files:
        *   `mv ../local-agent/Dockerfile ./apps/backends/adk/`
        *   `mv ../local-agent/.dockerignore ./apps/backends/adk/`
        *   `mv ../local-agent/pyproject.toml ./apps/backends/adk/`
        *   `mv ../local-agent/uv.lock ./apps/backends/adk/`
        *   `mv ../local-agent/.env ./apps/backends/adk/`
        *   `mv ../local-agent/entrypoint.sh ./apps/backends/adk/`

6.  **Move Root Files:**
    *   Move Docker Compose files: `mv ../local-agent/docker-compose*.yml ./`
    *   Move main README: `mv ../local-agent/README.md ./`
    *   Move main gitignore: `mv ../local-agent/.gitignore ./`

**Phase 2: Update Code & Configuration**

7.  **Create `__init__.py` Files:**
    *   Ensure necessary `__init__.py` files exist:
        *   `touch packages/agents/__init__.py`
        *   `touch packages/agents/core/__init__.py`
        *   `touch packages/agents/core/prompts/__init__.py`
        *   `touch packages/agents/core/agents/__init__.py`
        *   `touch packages/agents/tools/__init__.py`
        *   `touch packages/agents/tools/search/__init__.py`
        *   `touch packages/api-schemas/__init__.py`

8.  **Create Package `pyproject.toml` Files:**
    *   Create minimal `pyproject.toml` files for each package. Example for `packages/agents/core/pyproject.toml`:
        ```toml
        [build-system]
        requires = ["setuptools>=61.0"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "monorepo-agents-core" # Unique name
        version = "0.1.0"
        dependencies = []
        [tool.setuptools.packages.find]
        where = ["."]
        ```
    *   Create similar files for `packages/agents/tools/pyproject.toml` and `packages/api-schemas/pyproject.toml`, adjusting names and dependencies.

9.  **Update Python Imports:**
    *   **`apps/backends/adk/agent.py`:**
        *   Change `from .sub_agents import ...` to `from packages.agents.core.agents import ...`
        *   Change `from .prompt import ...` to `from packages.agents.core.prompts.local_agent_prompt import local_agent_prompt`
        *   Change `from .tools import ...` to `from packages.agents.tools.filesystem import ...` and `from packages.agents.tools.search import ...`
        *   Relative imports for `.callbacks` remain.
    *   **`apps/backends/adk/api.py`:**
        *   Review `agent_dir` logic based on how `get_fast_api_app` loads agents.
    *   **`apps/backends/adk/callbacks/`:**
        *   Review imports; update any referencing old `tools` or `sub_agents` paths to use `packages.*`.
    *   **`packages/agents/core/agents/`:**
        *   Update imports to use new `packages.*` paths.
    *   **`packages/agents/tools/`:**
        *   Update imports if necessary.

10. **Configure ADK App Dependencies (`apps/backends/adk/pyproject.toml`):**
    *   Add dependencies on local packages using relative paths or editable installs. Example using editable installs (managed via `uv pip install -e ...`):
        ```toml
        [project]
        # ... other project info ...
        dependencies = [
            "google-adk", # Or specific ADK package
            "fastapi",
            # ... other direct deps ...
            # Local packages will be installed via editable mode
        ]
        ```
    *   *(Agent performing task will need a tool to run `uv pip install -e ../../packages/agents/core -e ../../packages/agents/tools -e ../../packages/api-schemas`)*

11. **Update Dockerfile (`apps/backends/adk/Dockerfile`):**
    *   Ensure `WORKDIR` is `/app`.
    *   Adjust `COPY` commands. If using editable installs via volumes, you might only copy `pyproject.toml` and `uv.lock` initially: `COPY pyproject.toml uv.lock* ./`
    *   Run `uv sync` or `uv pip install ...` (including editable installs if managed this way).
    *   Adjust `CMD` or `ENTRYPOINT` if necessary.

12. **Update Docker Compose (`./docker-compose.yml`):**
    *   Update the ADK service:
        *   Set `build.context` to `./apps/backends/adk`.
        *   Adjust `volumes`:
            *   `./apps/backends/adk:/app`
            *   `./packages:/packages` # Mount shared packages
            *   `~/Solutions:/repos` # Keep REPO_ROOT mapping
        *   Set `working_dir: /app`
        *   Set/Verify `PYTHONPATH=/app:/packages` environment variable.
        *   Ensure `command` runs `uvicorn api:app ...`.

13. **Update Run Scripts (`apps/backends/adk/entrypoint.sh`):**
    *   Ensure paths and commands are correct relative to `/app`. Check `PYTHONPATH`.

**Phase 3: Testing & Cleanup**

14. **Install & Test:**
    *   Locally: `cd apps/backends/adk`, run `uv venv`, `uv pip install -e ../../packages/...`, `uv run uvicorn api:app ...`. Test API.
    *   Docker: From `monorepo-root`, run `docker compose build && docker compose up`. Test API.

15. **Cleanup:**
    *   Delete the original `../local-agent` directory once verified.

