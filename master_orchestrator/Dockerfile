# Master Orchestrator Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Create application directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    ORCHESTRATOR_ENVIRONMENT=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r orchestrator && useradd -r -g orchestrator orchestrator

# Create application directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY master_orchestrator/ master_orchestrator/
COPY start.py .
COPY config.example.yaml .

# Create required directories
RUN mkdir -p data logs && \
    chown -R orchestrator:orchestrator /app

# Switch to non-root user
USER orchestrator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "start.py", "--config", "config.example.yaml"]

# Labels
LABEL maintainer="Master Orchestrator Team" \
      version="0.1.0" \
      description="Agentic Multi-Project Orchestration System" \
      org.opencontainers.image.title="Master Orchestrator" \
      org.opencontainers.image.description="Enterprise-scale AI orchestration platform" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Master Orchestrator Project"