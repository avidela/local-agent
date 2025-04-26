import os
import logging
from dotenv import load_dotenv
from google.adk.cli.fast_api import get_fast_api_app

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

agent_dir = os.getenv("AGENT_DIR")

if not agent_dir:
    logger.warning("AGENT_DIR not found in .env file.")
    agent_dir = "agents/adk"

app = get_fast_api_app(
    agent_dir=agent_dir,
    allow_origins=["*"],
    web=True
)

logger.info("FastAPI app created. Run with: uvicorn api:app --reload --port 8001")