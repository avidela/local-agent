# Stage 1: Install uv and application dependencies
FROM python:3.13-alpine AS builder

# Copy uv from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the necessary dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (without --locked flag to allow updates)
RUN uv sync

# Stage 2: Final image
FROM python:3.13-alpine

# Copy uv from the builder stage (where it was copied from ghcr.io/astral-sh/uv:latest)
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY apps/backends/adk /app/apps/backends/adk

# Expose the port your application listens on
EXPOSE 8001

# Command to run your application
CMD ["uv","run","uvicorn", "apps.backends.adk.api:app", "--host", "0.0.0.0", "--port", "8001"]
