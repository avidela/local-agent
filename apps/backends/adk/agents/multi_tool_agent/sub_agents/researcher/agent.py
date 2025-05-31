from google.genai.types import GenerateContentConfig
from google.adk.agents import Agent
from ...tools import google_search_agent_tool
from .prompt import INSTRUCTION_PROMPT


root_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash-exp",
    instruction= INSTRUCTION_PROMPT,
    tools=[google_search_agent_tool],
    generate_content_config= GenerateContentConfig(temperature=0.10),
)