# Stage 1: Build dependencies and install packages
FROM python:3.11-slim AS builder

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=5

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.5

WORKDIR /app

# Copy only dependency files
COPY pyproject.toml poetry.lock /app/

# Export dependencies and install via pip
RUN poetry export --without-hashes --only main -f requirements.txt > requirements.txt \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Clean up build dependencies and caches
RUN apt-get remove -y build-essential libpq-dev curl && apt-get autoremove -y && apt-get clean && \
    rm -rf /root/.cache \
           /root/.local/share/pypoetry \
           /var/lib/apt/lists/* \
           /usr/share/doc \
           /usr/share/man \
           /usr/share/locale \
           /tmp/*

# Stage 2: Runtime environment
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install only runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Expose API port
EXPOSE 8000

# Start application directly
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
