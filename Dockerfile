FROM python:3.13-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache postgresql-libs curl && \
    apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev

# Install UV
RUN curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh && \
    uv --version

# Copy project metadata
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    uv sync --locked && \
    echo "Virtual environment contents:" && \
    ls -la .venv/bin/ && \
    # Test that Python is working
    .venv/bin/python --version && \
    # Test import of a dependency
    .venv/bin/python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')" && \
    # Clean up build dependencies to reduce image size
    apk --purge del .build-deps

# Copy the rest of the code
COPY . .

# Set up virtual environment for runtime
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 8001

# Run directly with the Python from the virtual environment
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]