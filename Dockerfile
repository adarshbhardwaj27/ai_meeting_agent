# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# System dependencies (ffmpeg for ElevenLabs TTS audio output)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r meetmind && useradd -r -g meetmind meetmind

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/

# Logs and data directories (writable by app user)
RUN mkdir -p logs data && chown -R meetmind:meetmind /app

USER meetmind

# WebSocket port (Recall.ai realtime transcript) + API port
EXPOSE 8000 8766

# Uvicorn — single worker is fine for MVP; use gunicorn+uvicorn for multi-worker
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--no-access-log"]
