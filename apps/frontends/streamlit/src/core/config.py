import os

ADK_BACKEND_URL = os.environ.get("ADK_BACKEND_URL", "http://localhost:8001")
# Note: Agent name (app_name for ADK) is now dynamically selected by the user.
# This config file only holds truly static, global application settings.