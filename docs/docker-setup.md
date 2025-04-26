# Docker Setup for Local Agent

This document explains how to run the Local Agent project in Docker using UV for Python package management.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Configuration

1. Create a `docker.env` file with your configuration:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```

2. Update the `docker-compose.yml` file if you need to expose any ports or modify volume mounts.

## Building and Running

### Build the Docker image

```bash
docker-compose build
```

### Run the container

```bash
docker-compose up
```

Or, to run in detached mode:

```bash
docker-compose up -d
```

### Stop the container

```bash
docker-compose down
```

## Development with Docker

The Docker setup mounts the local directory to `/app` in the container, allowing for changes to be reflected immediately without rebuilding the image. This is ideal for development.

## Python Package Management with UV, pyproject.toml and uv.lock

This Docker setup combines the modern `pyproject.toml` standard with [UV](https://github.com/astral-sh/uv), an extremely fast Python package manager and resolver. The build process:

1. Installs UV directly from the official installer
2. Uses UV sync with the uv.lock file to create a reproducible environment
3. Creates a virtual environment (.venv) in the container

UV offers several advantages over traditional package managers:
- Up to 10-100x faster installations
- Improved dependency resolution with precise lockfiles
- Lower memory usage
- Support for modern Python packaging standards

To add or update dependencies:

1. Update the `dependencies` list in `pyproject.toml`
2. Regenerate the lock file locally:
   ```bash
   uv lock
   ```
3. Sync your local environment to test changes:
   ```bash
   uv sync
   ```
4. Commit both pyproject.toml and uv.lock
5. Rebuild the Docker image:
   ```bash
   docker-compose build
   ```

For development dependencies that shouldn't be included in the Docker image, add them to the `[project.optional-dependencies]` section under `dev`.

## Troubleshooting

### API Key Issues

If you see authentication errors, make sure your `GOOGLE_API_KEY` is correctly set in the `docker.env` file and is valid.

### Container Not Starting

Check the logs for more information:

```bash
docker-compose logs
```

### Permission Issues

The container runs as a non-root user for security. If you're experiencing permission issues with mounted volumes, you may need to adjust permissions on your host machine.