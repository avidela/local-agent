# Use a lightweight Python base image
FROM python:3.13-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency management files
COPY pyproject.toml uv.lock ./

# Install dependencies directly (no virtual environment)
RUN uv pip install --system streamlit && \
    uv pip install --system -e .

# Verify streamlit is installed and in PATH
RUN which streamlit

# Copy the Streamlit application
COPY . .

# Set PYTHONPATH to include the app directory so Python can find the src module
ENV PYTHONPATH=/app

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application with the correct path
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
