FROM python:3.13-alpine

# Copy uv from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the necessary dependency files for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --locked

# Copy the rest of the application code
COPY . .

# Expose the port your application listens on
EXPOSE 8001

# Set PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Command to run your application
CMD ["uv","run","uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]