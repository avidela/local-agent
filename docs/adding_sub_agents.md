# Adding New Sub-agents

The Local Agent is designed to be extensible by adding new specialized sub-agents. Each sub-agent can have its own specific instructions, model, and set of tools, allowing for a clear separation of concerns and capabilities within the multi-agent system.

## Steps to Add a New Sub-agent

1.  **Create a Directory for the New Sub-agent:** Under the `local-agent/agents/adk/multi_tool_agent/sub_agents/` directory, create a new directory for your sub-agent. Use a descriptive name (e.g., `analyst`, `coder`).

    ```bash
    # Example: Creating a directory for an 'analyst' sub-agent
    mkdir local-agent/agents/adk/multi_tool_agent/sub_agents/analyst
    ```

2.  **Create `__init__.py`, `agent.py`, and `prompt.py` in the New Sub-agent Directory:**

    *   `__init__.py`: This file can be empty or contain necessary imports for the sub-agent's package.
    *   `agent.py`: This file will define the new sub-agent using the `Agent` class from `google.adk.agents`. You will configure its name, model, instruction prompt, and any specific tools it should have.
    *   `prompt.py`: This file will contain the detailed instruction prompt for the new sub-agent.

3.  **Define the Sub-agent in `agent.py`:** In the new sub-agent's `agent.py` file, define the agent using the `Agent` class. Import the instruction prompt from `prompt.py` and list any tools specific to this sub-agent.

    ```python
    # local-agent/agents/adk/multi_tool_agent/sub_agents/analyst/agent.py

    from google.adk.agents import Agent
    from google.genai.types import GenerateContentConfig
    # Import any tools specific to this agent
    # from ...tools import my_analyst_tool
    from .prompt import ANALYST_PROMPT

    root_agent = Agent(
        name="analyst",
        model="gemini-2.5-pro-exp-03-25", # Choose an appropriate model
        instruction=ANALYST_PROMPT,
        tools=[ # List tools specific to the analyst agent
            # my_analyst_tool,
        ],
        generate_content_config= GenerateContentConfig(temperature=0.10)
    )
    ```

4.  **Write the Sub-agent's Prompt in `prompt.py`:** In the new sub-agent's `prompt.py` file, define the `INSTRUCTION_PROMPT` (or a similar variable) that clearly describes the sub-agent's role, responsibilities, and how it should interact.

    ```python
    # local-agent/agents/adk/multi_tool_agent/sub_agents/analyst/prompt.py

    ANALYST_PROMPT ="""
    You are a data analyst agent. Your role is to analyze data provided by the user or accessed via your tools and provide insights and summaries.

    When a user asks for data analysis, you should:
    1. Understand the data source and the analysis required.
    2. Use your available tools to access and process the data.
    3. Perform the requested analysis.
    4. Summarize your findings clearly and concisely.
    5. Present the results to the user.
    6. Transfer back to the main agent when the analysis is complete.
    """
    ```

5.  **Import the New Sub-agent in the Main Agent's `__init__.py`:** In `local-agent/agents/adk/multi_tool_agent/sub_agents/__init__.py`, import the new sub-agent's `root_agent` instance.

    ```python
    # local-agent/agents/adk/multi_tool_agent/sub_agents/__init__.py

    from .researcher.agent import root_agent as researcher_agent
    from .developer.agent import root_agent as developer_agent
    from .analyst.agent import root_agent as analyst_agent # Import the new sub-agent
    ```

6.  **Add the New Sub-agent to the Main Agent's Sub-agent List:** In the main agent's `agent.py` file (`local-agent/agents/adk/multi_tool_agent/agent.py`), add the new sub-agent instance to the `sub_agents` list.

    ```python
    root_agent = Agent(
        # ... other agent configuration
        sub_agents=[researcher_agent, developer_agent, analyst_agent], # Add the new sub-agent here
        # ... other agent configuration
    )
    ```

7.  **Update the Main Agent's Prompt:** Modify the main agent's instruction prompt (`local-agent/agents/adk/multi_tool_agent/prompt.py`) to inform it about the new sub-agent and when to delegate tasks to it.

    ```python
    local_agent_prompt = """
    ...

    - If the request involves data analysis, transfer to the `analyst` agent.

    ...
    """
    ```

8.  **Test the New Sub-agent:** Run the agent and test the new sub-agent by giving it requests that should trigger delegation to it. Verify that the task is correctly delegated and handled by the new sub-agent.

By following these steps, you can successfully add new sub-agents to expand the capabilities and modularity of the Local Agent system.
