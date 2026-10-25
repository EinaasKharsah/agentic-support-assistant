FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Install dependencies first: this layer is reused when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model during the build, so the container starts fast
# and does not need the internet at run time.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"
# From now on, only use the downloaded copy (no internet calls at start-up).
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY data ./data

# Run as a normal user, not root. The feedback folder is the only place the app writes.
RUN useradd --create-home appuser \
    && mkdir -p /app/feedback \
    && chown -R appuser /app/feedback /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
