FROM python:3.12-slim

# Logs appear immediately in `docker compose logs`; no .pyc clutter
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
    
WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first (Docker layer caching)
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .

# Switch to non-root user before installing deps (so .venv is owned by appuser)
USER appuser

# Install dependencies only (production). --no-install-project because the
# app code is not copied yet, so this layer stays cached until deps change.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY --chown=appuser:appuser app/ app/

EXPOSE 8000

# python:3.12-slim has no curl, so the check uses Python's standard library
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

# --forwarded-allow-ips "*": trust X-Forwarded-For from nginx so rate limiting sees the real
# client IP. Safe ONLY because port 8000 is not published to the host (see docker-compose.yml).
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]