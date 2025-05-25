from datetime import datetime
import json
import os

from dotenv import load_dotenv
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.auth import AuthConfig
from google.adk.auth import AuthCredential
from google.adk.auth import AuthCredentialTypes
from google.adk.auth import OAuth2Auth
from google.adk.tools import ToolContext
from google.adk.tools.google_api_tool import CalendarToolset
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv()

oauth_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
oauth_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")

if not oauth_client_id:
    raise ValueError(
        "GOOGLE_OAUTH_CLIENT_ID environment variable not set. Please set it in your .env file."
    )
if not oauth_client_secret:
    raise ValueError(
        "GOOGLE_OAUTH_CLIENT_SECRET environment variable not set. Please set it in your .env file."
    )

SCOPES = ["https://www.googleapis.com/auth/calendar"]

calendar_toolset = CalendarToolset(
    client_id=oauth_client_id,
    client_secret=oauth_client_secret,
    tool_filter=["calendar_events_get","calendar_events_list"],
)

def update_time(callback_context: CallbackContext):
  now = datetime.now()
  formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
  callback_context.state["_time"] = formatted_time


root_agent = Agent(
    model="gemini-2.0-flash",
    name="calendar_agent",
    instruction="""
      You are a helpful personal calendar assistant.
      Use the provided tools to search for calendar events (use 10 as limit if user does't specify), and update them.
      Use "primary" as the calendarId if users don't specify.

      Scenario1:
      The user want to query the calendar events.
      Use list_calendar_events to search for calendar events.


      Scenario2:
      User want to know the details of one of the listed calendar events.
      Use get_calendar_event to get the details of a calendar event.


      Current user:
      <User>
      {userInfo?}
      </User>

      Currnet time: {_time}
""",
    tools=[calendar_toolset],
    before_agent_callback=update_time,
)