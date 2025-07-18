from google.genai.types import GenerateContentConfig
from google.adk.agents import Agent
from google.adk.tools.langchain_tool import LangchainTool
from .prompt import INSTRUCTION_PROMPT


root_agent = Agent(
    name="developer",
    model="gemini-2.0-flash-exp",
    instruction= INSTRUCTION_PROMPT,
    tools=[],
    generate_content_config= GenerateContentConfig(temperature=0.10),
)