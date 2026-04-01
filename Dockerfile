# server/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds the Cascade Containment environment server.
# Exposes port 7860 — required for Hugging Face Spaces deployment.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY constants.py .
COPY server/ ./server/
COPY core/ ./core/

ENV PYTHONPATH="/app:/app/server"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]