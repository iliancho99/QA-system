FROM python:3.11-slim

# - PYTHONUNBUFFERED: stream logs straight to the container output.
# - PYTHONDONTWRITEBYTECODE: skip .pyc files in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential is needed to compile a few native deps (e.g. chromadb's
# hnswlib) when no prebuilt wheel matches; remove the apt lists to stay slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then copy the application code.
COPY . .

EXPOSE 8000

# On startup: index the documents, then serve the API. Ingestion only needs the
# local embedding model (not the LLM), so it runs even before the model is pulled.
CMD ["sh", "-c", "python -m scripts.ingest && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
