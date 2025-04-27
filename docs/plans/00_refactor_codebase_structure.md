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
│   ├── plans/                # Directory for planned refactors/work  <-- Updated line
│   │   └── 00_refactor_codebase_structure.md # This file           <-- Updated line
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

