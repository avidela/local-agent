#!/bin/bash
set -e

# Print Python and UV versions for debugging
python --version
uv --version

# Check if the GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY is not set. Please set it in your .env file."
    exit 1
fi

# Run the application
exec python -m multi_tool_agent