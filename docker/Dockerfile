# Multi-stage Dockerfile for IntegratedML Pluggable Models
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    unixodbc-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: IRIS Python packages will be available when connected to IRIS container
# intersystems-iris and sqlalchemy-iris packages are not available on PyPI
# They are typically provided by the IRIS installation itself

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models/cache /app/logs && \
    chown -R appuser:appuser /app

# Application stage
FROM base as app

USER appuser

EXPOSE 8080

CMD ["python", "-m", "shared.database.setup_database"]

# Jupyter stage
FROM base as jupyter

# Install Jupyter and additional packages
RUN pip install --no-cache-dir \
    jupyterlab==4.0.* \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    notebook

USER appuser

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Development stage
FROM jupyter as dev

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit

USER appuser

CMD ["bash"]