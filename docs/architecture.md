# Agent Architecture

The Local Agent is designed as a multi-agent system using Google's Agent Development Kit (ADK). This architecture allows for the delegation of specific tasks to specialized sub-agents, making the system more modular and capable.

## Core Components

*   **`local_agent` (Main Agent):** This is the primary agent that receives user requests. It uses the `gemini-2.5-pro-exp-03-25` model with a temperature of `0.10`. In addition to its main instruction prompt (`prompt.py`), it receives a dynamic `global_instruction` containing the current date. It is responsible for understanding the user's intent and delegating the task to the most appropriate sub-agent or executing it directly using its available tools.

*   **Sub-agents:** The `local_agent` can delegate tasks to specialized sub-agents. Currently, the system includes:
    *   **`researcher` Agent:** Specialized in finding and critically evaluating *online* information using tools like Google Search. Its detailed prompt guides it to understand requests, gather information from diverse online sources, assess source credibility (authority, currency, accuracy, purpose), compare sources, identify bias, synthesize findings, and provide structured, evidence-based responses before transferring back to the `local_agent`.
    *   **`developer` Agent:** (Currently has an empty prompt and no specific tools defined in its agent file, suggesting its role might be intended for future expansion or handled by the main agent's tools). Its intended purpose is likely related to code manipulation, analysis, or execution.

*   **Tools:** Agents interact with the environment and perform actions through tools. The `local_agent` is configured with a suite of tools:
    *   **Standard Filesystem Tools (via `LangchainTool` wrapper):** `FileSearchTool`, `ListDirectoryTool`, `ReadFileTool`, `WriteFileTool`, `CopyFileTool`, `MoveFileTool`, `DeleteFileTool`. These operate relative to the configured `REPO_ROOT`.
    *   **Custom Filesystem Tools (via `FunctionTool` wrapper):** `grep_file`, `find_files`, `diff_files`. These wrap the corresponding system commands (`grep`, `find`, `diff`) and also operate relative to `REPO_ROOT`.
    Sub-agents can also have their own specific tools (e.g., the `researcher` uses a Google Search tool).

## Task Flow

When a user submits a request to the `local_agent`:

1.  The `local_agent` processes the request based on its main instruction prompt (`prompt.py`) and the global instruction.
2.  Based on the prompt's logic and the nature of the request, the `local_agent` decides whether to:
    *   Handle the request directly using its configured tools.
    *   Delegate the request to a relevant sub-agent (e.g., if the request requires online research and evaluation, it transfers to the `researcher`).
3.  If a sub-agent handles the request, it uses its own instructions and tools to fulfill the task.
4.  After a sub-agent completes its task (or if the `local_agent` handled it directly), the result is typically returned, and the `local_agent` may format the final response to the user. Callbacks may intercept or modify input/output at various stages (see [Agent Callbacks](callbacks.md)).

## Extending the Architecture

The modular design allows for adding new capabilities by:

*   Creating new specialized tools (see [Adding New Tools](adding_tools.md)).
*   Developing new sub-agents for distinct types of tasks (see [Adding New Sub-agents](adding_sub_agents.md)).
*   Modifying the main agent's prompt to improve its task routing and handling logic.
*   Implementing or modifying callback handlers (see [Agent Callbacks](callbacks.md)).

