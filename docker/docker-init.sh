#!/bin/bash

# IntegratedML Pluggable Models Docker Initialization Script
# This script sets up the necessary directory structure and permissions

set -e

echo "Initializing IntegratedML Pluggable Models Docker environment..."

# Create necessary directories
mkdir -p docker/volumes/iris_data
mkdir -p docker/volumes/iml_data
mkdir -p docker/volumes/model_cache
mkdir -p docker/iris-config
mkdir -p data/credit_risk
mkdir -p data/fraud_detection
mkdir -p data/sales_forecasting
mkdir -p models/cache
mkdir -p logs

# Set appropriate permissions
chmod 755 docker/volumes/iris_data
chmod 755 docker/volumes/iml_data
chmod 755 docker/volumes/model_cache
chmod 755 data
chmod 755 models
chmod 755 logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your specific configuration values."
fi

# Ensure Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

echo "Docker environment initialized successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: docker-compose up --build"
echo "3. Access IRIS Management Portal at: http://localhost:52773/csp/sys/UtilHome.csp"
echo "4. Access Jupyter Lab at: http://localhost:8888"
echo ""