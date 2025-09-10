# Docker Setup Guide for IntegratedML Flexible Model Integration

This guide provides step-by-step instructions for setting up the complete development environment using Docker with IRIS Community Edition and IntegratedML.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for containers
- 20GB free disk space

## Quick Start

### 1. Initialize Environment

```bash
# Clone the repository
git clone <repository-url>
cd flexible_model_integration

# Initialize Docker environment
chmod +x docker/docker-init.sh
./docker/docker-init.sh
```

### 2. Configure Environment

```bash
# Copy and edit environment configuration
cp .env.example .env
# Edit .env with your preferred settings
```

Key configuration options in `.env`:
```bash
# Database Configuration
IRIS_USERNAME=demo
IRIS_PASSWORD=demo
IRIS_NAMESPACE=USER

# Application Ports
IRIS_PORT=1972
IRIS_WEB_PORT=52773
IML_APP_PORT=8080
JUPYTER_PORT=8888

# Demo Data Size
CREDIT_RISK_SAMPLES=10000
FRAUD_DETECTION_SAMPLES=50000
SALES_FORECASTING_DAYS=365
```

### 3. Start Services

```bash
# Start all services
docker-compose up --build -d

# Check service status
docker-compose ps
```

### 4. Verify Installation

```bash
# Check IRIS is running
docker-compose logs iris

# Test database connection
python -c "from shared.database import test_connection; print('✅ Success' if test_connection() else '❌ Failed')"
```

## Service Access

Once all containers are running, you can access:

- **IRIS Management Portal**: http://localhost:52773/csp/sys/UtilHome.csp
  - Username: `demo`
  - Password: `demo`
  
- **Jupyter Lab**: http://localhost:8888
  - No password required (development mode)
  
- **Application API**: http://localhost:8080
  - Health check: http://localhost:8080/health

## Database Setup

### Automatic Setup

Run the database setup script to initialize schemas and load demo data:

```bash
# Setup database and load all demo data
python shared/database/setup_database.py

# Or setup specific demo data
python shared/database/data_loader.py
```

### Manual Setup

Connect to IRIS and run the initialization scripts located in `docker/iris-init/`. The main script is [`01_setup_integratedml.sql`](../docker/iris-init/01_setup_integratedml.sql), with additional scripts for populating data.

## Running Demos

### Credit Risk Assessment Demo

```bash
python run_credit_risk_demo.py
```

This demo will:
- Connect to IRIS database
- Load credit risk data
- Train classification model
- Make real-time risk predictions
- Show model evaluation metrics

### Fraud Detection Demo

```bash
python run_fraud_detection_demo.py
```

This demo will:
- Load transaction data
- Train ensemble fraud detection model
- Simulate real-time fraud detection
- Analyze historical patterns
- Show monitoring dashboard

### Sales Forecasting Demo

```bash
python run_sales_forecasting_demo.py
```

This demo will:
- Load sales historical data
- Train regression forecasting model
- Generate sales forecasts
- Perform seasonal analysis
- Show business intelligence metrics

## Docker Services

### IRIS Database Container

```yaml
services:
  iris:
    image: intersystemsdc/iris-community:latest
    ports:
      - "1972:1972"    # Database port
      - "52773:52773"  # Management portal
    volumes:
      - iris_data:/opt/irisapp/data
      - ./docker/iris-init:/opt/irisapp/init:ro
```

**Features:**
- IRIS Community Edition
- IntegratedML enabled
- Persistent data storage
- Initialization scripts
- Health checks

### Application Container

```yaml
services:
  iml_app:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      iris:
        condition: service_healthy
```

**Features:**
- Python 3.11 environment
- All required packages installed
- Database connection utilities
- Model management tools
- Health monitoring

### Jupyter Container

```yaml
services:
  jupyter:
    build: 
      target: jupyter
    ports:
      - "8888:8888"
```

**Features:**
- JupyterLab environment
- Pre-installed ML libraries
- Access to all project code
- Interactive notebooks
- Visualization tools

## Troubleshooting

### Common Issues

#### 1. Container Startup Failures

```bash
# Check container logs
docker-compose logs iris
docker-compose logs iml_app

# Restart services
docker-compose down
docker-compose up --build
```

#### 2. Database Connection Issues

```bash
# Test IRIS connectivity
docker exec -it flexible_model_integration_iris iris session iris

# Check IRIS status
docker exec -it flexible_model_integration_iris iris status
```

#### 3. Memory Issues

```bash
# Check container resource usage
docker stats

# Increase Docker memory allocation in Docker Desktop
# Minimum 8GB recommended
```

#### 4. Port Conflicts

```bash
# Check port usage
netstat -tulpn | grep :1972
netstat -tulpn | grep :52773

# Modify ports in .env file if needed
IRIS_PORT=1973
IRIS_WEB_PORT=52774
```

### Performance Optimization

#### Database Performance

```bash
# Increase IRIS memory allocation
docker-compose up -d --scale iris=1 --memory=4g iris
```

#### Application Performance

```bash
# Monitor application performance
docker-compose exec iml_app python -c "
from shared.database import get_connection
conn = get_connection()
info = conn.get_integratedml_info()
print(f'Models: {info.get(\"model_count\", 0)}')
"
```

## Development Workflow

### 1. Code Changes

```bash
# For Python code changes, restart application
docker-compose restart iml_app

# For database schema changes
docker-compose exec iris iris session iris
# Run SQL commands or scripts
```

### 2. Model Development

```bash
# Access Jupyter for interactive development
# Open: http://localhost:8888

# Or use Python directly
docker-compose exec iml_app python
```

### 3. Testing

```bash
# Run all tests
docker-compose exec iml_app python -m pytest

# Run specific demo tests
docker-compose exec iml_app python run_credit_risk_demo.py
```

## Data Persistence

### Volume Management

```bash
# View volumes
docker volume ls | grep flexible_model_integration

# Backup data
docker-compose exec iris iris backup

# Restore data
docker-compose exec iris iris restore <backup-file>
```

### Data Reset

```bash
# Reset all data
docker-compose down -v
docker-compose up --build

# Reset specific demo data
python shared/database/data_loader.py
```

## Production Considerations

### Security

- Change default passwords in production
- Use environment-specific `.env` files
- Enable IRIS SSL/TLS
- Implement proper authentication

### Monitoring

- Enable health checks
- Set up log aggregation
- Monitor resource usage
- Configure alerts

### Scaling

- Use Docker Swarm or Kubernetes
- Implement load balancing
- Configure backup strategies
- Set up monitoring dashboards

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review container logs: `docker-compose logs`
3. Verify environment configuration in `.env`
4. Test database connectivity with provided utilities
5. Check IRIS documentation for IntegratedML specifics

## Next Steps

After successful setup:

1. Explore the interactive Jupyter notebooks
2. Run all three demo scripts
3. Modify model parameters and retrain
4. Develop custom models using the flexible framework
5. Integrate with your own data sources