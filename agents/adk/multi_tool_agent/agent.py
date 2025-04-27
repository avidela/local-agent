import os
from datetime import date
from google.adk.agents import Agent
from google.genai.types import GenerateContentConfig
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.tools import FunctionTool # Import FunctionTool
from langchain_community.tools.file_management.file_search import FileSearchTool
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.file_management.copy import CopyFileTool
from langchain_community.tools.file_management.move import MoveFileTool
from langchain_community.tools.file_management.delete import DeleteFileTool
from .sub_agents import researcher_agent, developer_agent
from .prompt import local_agent_prompt
# Import the new grep_file function
from .tools.filesystem.grep_tool import grep_file

root_dir_raw = os.getenv("REPO_ROOT", os.getcwd())
root_dir = os.path.expanduser(root_dir_raw)

root_agent = Agent(
    name="local_agent",
    # model="gemini-2.0-flash-exp",
    # model="gemini-2.5-pro-exp-03-25",
    model="gemini-2.5-flash-preview-04-17",
    instruction=local_agent_prompt,
    global_instruction=lambda ctx: (
        f"""
        You are a Software Development Multi Agent System.
        Today's date: {date.today()}
        """
    ),
    sub_agents=[researcher_agent,developer_agent],
    tools=[
        LangchainTool(tool=FileSearchTool(root_dir=root_dir)),
        LangchainTool(tool=ListDirectoryTool(root_dir=root_dir)),
        LangchainTool(tool=ReadFileTool(root_dir=root_dir)),
        LangchainTool(tool=WriteFileTool(root_dir=root_dir)),
        LangchainTool(tool=CopyFileTool(root_dir=root_dir)),
        LangchainTool(tool=MoveFileTool(root_dir=root_dir)),
        LangchainTool(tool=DeleteFileTool(root_dir=root_dir)),
        # Add the new grep tool here using FunctionTool.
        FunctionTool(func=grep_file)
    ],
    generate_content_config= GenerateContentConfig(temperature=0.10)
)
