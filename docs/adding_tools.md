# Adding New Tools

The Local Agent's capabilities can be extended by adding new tools. Tools are functions or classes that the agent can call to perform specific actions, such as interacting with external APIs, accessing services, or performing computations.

The Agent Development Kit (ADK) provides wrappers like `FunctionTool` and `LangchainTool` to easily integrate Python functions or existing Langchain tools.

## Steps to Add a New Tool

1.  **Implement the Tool Logic:** Write the Python code for your tool. This can be a simple function or a class with a method that performs the desired action. The function/method should accept parameters that the agent can provide and return a result that the agent can interpret.

    *   **For a simple function:** Define a Python function that takes arguments and returns a value.
    *   **For a Langchain Tool:** If you are wrapping an existing Langchain tool, ensure you have the necessary Langchain library installed and the tool is correctly initialized.

2.  **Choose the Appropriate ADK Wrapper:**

    *   Use `FunctionTool` to wrap a standard Python function.
    *   Use `LangchainTool` to wrap a Langchain `BaseTool` instance.

3.  **Instantiate the Tool:** In your agent's definition file (e.g., `local-agent/agents/adk/multi_tool_agent/agent.py`), import your tool implementation and instantiate the appropriate ADK wrapper.

    ```python
    # Example: Wrapping a simple function
    from .tools.my_new_tool import my_function
    from google.adk.tools import FunctionTool

    my_new_tool_instance = FunctionTool(func=my_function)

    # Example: Wrapping a Langchain tool
    # from langchain_community.tools import MyLangchainTool
    # from google.adk.tools import LangchainTool

    # my_langchain_tool_instance = LangchainTool(tool=MyLangchainTool())
    ```

4.  **Add the Tool to the Agent's Tool List:** Include the instantiated tool wrapper in the `tools` list when defining your agent (e.g., in the `Agent` constructor in `agent.py`).

    ```python
    root_agent = Agent(
        # ... other agent configuration
        tools=[
            # ... existing tools
            my_new_tool_instance,
            # my_langchain_tool_instance,
        ],
        # ... other agent configuration
    )
    ```

5.  **Update the Agent's Prompt:** Modify the agent's instruction prompt (e.g., in `local-agent/agents/adk/multi_tool_agent/prompt.py`) to inform the agent about the new tool. Clearly describe the tool's purpose, when it should be used, and what arguments it requires. This is crucial for the agent to be able to effectively use the new tool.

    ```python
    local_agent_prompt = """
    ...

    - When the user asks about [task the new tool performs], use the `my_new_tool_instance` tool.
      - This tool requires the following parameters: [parameter names and descriptions].

    ...
    """
    ```

6.  **Test the New Tool:** Run the agent and test the new tool by giving it requests that should trigger its usage. Verify that the tool is called correctly and produces the expected results.

By following these steps, you can successfully add new tools to enhance the capabilities of the Local Agent.
