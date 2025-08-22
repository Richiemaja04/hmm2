# Behavioral Biometrics Authentication Agent - Docker Configuration
# Multi-stage build for optimized production container

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=2.1.0
ARG COMMIT_SHA

# Add labels
LABEL maintainer="SecureAuth Team <team@secureauth.io>"
LABEL version="${VERSION}"
LABEL description="Behavioral Biometrics Authentication Agent"
LABEL build-date="${BUILD_DATE}"
LABEL commit-sha="${COMMIT_SHA}"

# Set environment variables for build
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

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip cache purge

# Production stage
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    FLASK_DEBUG=False \
    PORT=5000

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create production configuration
RUN echo "# Production Configuration" > .env \
    && echo "SECRET_KEY=\${SECRET_KEY}" >> .env \
    && echo "JWT_SECRET_KEY=\${JWT_SECRET_KEY}" >> .env \
    && echo "FLASK_ENV=production" >> .env \
    && echo "FLASK_DEBUG=False" >> .env \
    && echo "DATABASE_URL=\${DATABASE_URL:-sqlite:///data/behavioral_auth.db}" >> .env

# Initialize database schema
RUN python -c "from database.db_manager import DatabaseManager; from config import Config; db = DatabaseManager('sqlite:///data/behavioral_auth.db'); db.init_database()" || true

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command
CMD ["python", "-m", "gunicorn", "--worker-class", "eventlet", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "60", "--keepalive", "5", "--max-requests", "1000", "--max-requests-jitter", "50", "app:app"]

# Development stage
FROM production as development

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=True

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    pytest \
    pytest-flask \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Create development configuration
RUN echo "# Development Configuration" > .env.dev \
    && echo "SECRET_KEY=dev-secret-key-not-for-production" >> .env.dev \
    && echo "JWT_SECRET_KEY=dev-jwt-key-not-for-production" >> .env.dev \
    && echo "FLASK_ENV=development" >> .env.dev \
    && echo "FLASK_DEBUG=True" >> .env.dev \
    && echo "DATABASE_URL=sqlite:///data/behavioral_auth_dev.db" >> .env.dev

# Switch back to appuser
USER appuser

# Development command
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "5000"]

# Testing stage
FROM development as testing

# Switch to root for test setup
USER root

# Copy test files
COPY tests/ tests/
COPY pytest.ini .
COPY .coveragerc .

# Install additional test dependencies
RUN pip install \
    pytest-xdist \
    pytest-mock \
    coverage[toml] \
    factory-boy

# Set up test environment
RUN echo "# Test Configuration" > .env.test \
    && echo "SECRET_KEY=test-secret-key" >> .env.test \
    && echo "JWT_SECRET_KEY=test-jwt-key" >> .env.test \
    && echo "FLASK_ENV=testing" >> .env.test \
    && echo "TESTING=True" >> .env.test \
    && echo "DATABASE_URL=sqlite:///:memory:" >> .env.test

# Create test database
RUN python -c "import os; os.environ['TESTING']='True'; from app import db_manager; db_manager.init_database()" || true

# Change ownership back to appuser
RUN chown -R appuser:appuser /app

# Switch back to appuser
USER appuser

# Test command
CMD ["python", "-m", "pytest", "--cov=.", "--cov-report=html", "--cov-report=term", "-v"]

# Production-ready stage with additional optimizations
FROM production as optimized

# Switch to root for optimizations
USER root

# Install production optimizations
RUN pip install \
    gevent \
    psycopg2-binary \
    redis \
    sentry-sdk[flask]

# Optimize Python
RUN python -m compileall -b . \
    && find . -name "*.py" -delete \
    && find . -name "__pycache__" -exec rm -rf {} + || true

# Create startup script
RUN echo '#!/bin/bash' > /app/start.sh \
    && echo 'set -e' >> /app/start.sh \
    && echo '' >> /app/start.sh \
    && echo '# Initialize database if needed' >> /app/start.sh \
    && echo 'python -c "from database.db_manager import DatabaseManager; from config import Config; db = DatabaseManager(Config.DATABASE_URL); db.init_database()" || true' >> /app/start.sh \
    && echo '' >> /app/start.sh \
    && echo '# Start application' >> /app/start.sh \
    && echo 'exec gunicorn \' >> /app/start.sh \
    && echo '    --worker-class gevent \' >> /app/start.sh \
    && echo '    --workers ${WORKERS:-2} \' >> /app/start.sh \
    && echo '    --worker-connections ${WORKER_CONNECTIONS:-1000} \' >> /app/start.sh \
    && echo '    --bind 0.0.0.0:${PORT:-5000} \' >> /app/start.sh \
    && echo '    --timeout ${TIMEOUT:-120} \' >> /app/start.sh \
    && echo '    --keepalive ${KEEPALIVE:-5} \' >> /app/start.sh \
    && echo '    --max-requests ${MAX_REQUESTS:-1000} \' >> /app/start.sh \
    && echo '    --max-requests-jitter ${MAX_REQUESTS_JITTER:-50} \' >> /app/start.sh \
    && echo '    --preload \' >> /app/start.sh \
    && echo '    --access-logfile - \' >> /app/start.sh \
    && echo '    --error-logfile - \' >> /app/start.sh \
    && echo '    --log-level ${LOG_LEVEL:-info} \' >> /app/start.sh \
    && echo '    app:app' >> /app/start.sh \
    && chmod +x /app/start.sh

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch back to appuser
USER appuser

# Optimized startup
CMD ["/app/start.sh"]

# Default target is production
FROM production