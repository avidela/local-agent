# Multi-stage Dockerfile for PydanticAI Agents Service
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Development stage
FROM base AS development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --dev

# Copy source code
COPY . .

# Create migration entrypoint script
RUN echo '#!/bin/bash\nset -e\necho "Running database migrations..."\nuv run alembic upgrade head\necho "Starting application..."\nexec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint that runs migrations first
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (production only)
RUN uv sync --no-dev

# Copy source code
COPY . .

# Create migration entrypoint script
RUN echo '#!/bin/bash\nset -e\necho "Running database migrations..."\nuv run alembic upgrade head\necho "Starting application..."\nexec "$@"' > /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app && chown app:app /entrypoint.sh
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint that runs migrations first
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]