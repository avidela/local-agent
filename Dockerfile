FROM python:3.13-alpine
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /app

WORKDIR /app
RUN uv sync --locked

EXPOSE 8001

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uv","run","uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]