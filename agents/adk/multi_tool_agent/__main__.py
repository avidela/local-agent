#!/usr/bin/env python
"""Main entry point for the multi_tool_agent package."""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
required_vars = ["GOOGLE_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Import after environment is set up
from .agent import root_agent  # noqa

def main():
    """Run the agent."""
    logger.info("Starting Multi-Tool Agent")
    
    # Here you would typically start a server or interface to the agent
    # For now, just print a message indicating it's ready
    logger.info("Agent is ready!")
    
    # Keep the process running
    # In a real application, you might have a proper server loop here
    try:
        # This is a placeholder - replace with actual application logic
        while True:
            input_text = input("Enter a question (or 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break
                
            # Process the input with the agent
            response = root_agent.generate_content(input_text)
            
            # Print the response
            print("\nAgent response:")
            print(response)
            print("\n" + "-"*50 + "\n")
            
    except KeyboardInterrupt:
        logger.info("Shutting down on user request")
    
    logger.info("Multi-Tool Agent stopped")

if __name__ == "__main__":
    main()