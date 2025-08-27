# Deployment Guide

## 🚀 Production Deployment Strategies

This comprehensive guide covers deploying IntegratedML Flexible Model Integration in production environments, from single-server installations to large-scale distributed deployments with high availability and performance optimization.

---

## 📋 Table of Contents

1. [Deployment Architecture Overview](#deployment-architecture-overview)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Model Deployment Strategies](#model-deployment-strategies)
5. [Container Deployment](#container-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Performance Optimization](#performance-optimization)
8. [Security Configuration](#security-configuration)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [CI/CD Integration](#cicd-integration)
11. [Scaling Strategies](#scaling-strategies)
12. [Backup and Recovery](#backup-and-recovery)
13. [Troubleshooting](#troubleshooting)

---

## 🏗️ Deployment Architecture Overview

### Single-Server Deployment

Ideal for development, testing, and small-scale production environments.

```
┌─────────────────────────────────────┐
│            Single Server            │
├─────────────────────────────────────┤
│  ┌─────────────┐  ┌───────────────┐ │
│  │   IRIS DB   │  │  ML Models    │ │
│  │             │  │  - Credit     │ │
│  │             │  │  - Fraud      │ │
│  │             │  │  - Forecast   │ │
│  └─────────────┘  └───────────────┘ │
├─────────────────────────────────────┤
│         Application Layer           │
│  ┌─────────────┐  ┌───────────────┐ │
│  │   Web API   │  │  Batch Jobs   │ │
│  └─────────────┘  └───────────────┘ │
└─────────────────────────────────────┘
```

**Configuration:**
```yaml
# config/single_server.yaml
deployment:
  type: "single_server"
  instance_type: "standard"
  
database:
  host: "localhost"
  port: 1972
  namespace: "ML"
  
models:
  storage_path: "/opt/models"
  cache_size: "2GB"
  workers: 4
```

### High Availability Deployment

Production environment with redundancy and load balancing.

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └─────────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ IRIS    │          │ IRIS    │          │ IRIS    │
   │ Node 1  │◄────────►│ Node 2  │◄────────►│ Node 3  │
   │         │          │         │          │         │
   │ Models  │          │ Models  │          │ Models  │
   └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ App     │          │ App     │          │ App     │
   │ Server  │          │ Server  │          │ Server  │
   └─────────┘          └─────────┘          └─────────┘
```

**Configuration:**
```yaml
# config/ha_deployment.yaml
deployment:
  type: "high_availability"
  replicas: 3
  
load_balancer:
  type: "nginx"
  algorithm: "round_robin"
  health_check_interval: 30
  
database:
  cluster_config:
    mirror_async: true
    failover_timeout: 10
    data_servers: ["iris1", "iris2", "iris3"]
```

### Microservices Architecture

Enterprise deployment with service separation and independent scaling.

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                            │
└─────────────────┬───────────────┬───────────────────────────┘
                  │               │
    ┌─────────────▼─────────────┐ │ ┌─────────────────────────┐
    │   Credit Risk Service    │ │ │  Fraud Detection Service │
    │                          │ │ │                         │
    │ ┌──────────┐ ┌─────────┐ │ │ │ ┌──────────┐ ┌─────────┐ │
    │ │ ML Model │ │ IRIS DB │ │ │ │ │ ML Model │ │ IRIS DB │ │
    │ └──────────┘ └─────────┘ │ │ │ └──────────┘ └─────────┘ │
    └──────────────────────────┘ │ └─────────────────────────┘
                                 │
               ┌─────────────────▼─────────────────┐
               │     Sales Forecasting Service     │
               │                                   │
               │ ┌──────────┐ ┌─────────┐         │
               │ │ ML Model │ │ IRIS DB │         │
               │ └──────────┘ └─────────┘         │
               └───────────────────────────────────┘
```

---

## 🛠️ Environment Setup

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+, CentOS 8+)

#### Recommended Production
- **CPU**: 16+ cores, 3.0 GHz
- **RAM**: 32+ GB
- **Storage**: 500+ GB NVMe SSD
- **Network**: 10 Gbps
- **OS**: Linux with NUMA optimization

#### Large Scale Enterprise
- **CPU**: 32+ cores, 3.5 GHz
- **RAM**: 128+ GB
- **Storage**: 2+ TB NVMe SSD with RAID
- **Network**: 25+ Gbps with redundancy
- **OS**: Optimized Linux with real-time kernel

### Operating System Configuration

#### Ubuntu/Debian Setup
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.9 python3.9-dev python3.9-venv \
    build-essential cmake \
    libblas-dev liblapack-dev \
    libssl-dev libffi-dev \
    curl wget git htop

# Configure system limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* soft nproc 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nproc 65536" | sudo tee -a /etc/security/limits.conf

# Configure kernel parameters
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf
echo "net.core.somaxconn=65535" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### RHEL/CentOS Setup
```bash
# Enable EPEL and PowerTools
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled powertools

# Install dependencies
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    python39 python39-devel \
    cmake openblas-devel lapack-devel \
    openssl-devel libffi-devel \
    curl wget git htop

# Configure SELinux (if needed)
sudo setsebool -P httpd_can_network_connect 1
sudo setsebool -P httpd_execmem 1
```

### Python Environment Setup

```bash
# Create virtual environment
python3.9 -m venv /opt/integraedml-env
source /opt/integraedml-env/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# Install production dependencies
pip install \
    scikit-learn==1.3.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    lightgbm==4.0.0 \
    prophet==1.1.4 \
    pyyaml==6.0 \
    joblib==1.3.0 \
    psutil==5.9.5 \
    gunicorn==21.2.0 \
    uvicorn==0.23.0 \
    fastapi==0.103.0 \
    prometheus-client==0.17.0

# Install monitoring dependencies
pip install \
    structlog==23.1.0 \
    sentry-sdk==1.32.0 \
    statsd==4.0.1
```

---

## 🗄️ Database Configuration

### InterSystems IRIS Setup

#### Installation Configuration

```bash
# Download IRIS Community Edition
wget https://download.intersystems.com/download/iris/2024.1/IRIS-2024.1.0.267.0-lnxubuntux64.tar.gz

# Extract and install
tar -xzf IRIS-2024.1.0.267.0-lnxubuntux64.tar.gz
cd irisinstall_lnx

# Silent installation
sudo ./irisinstall \
    --installdir /opt/intersystems \
    --instance ML \
    --namespace ML \
    --superuser admin \
    --password YourSecurePassword123! \
    --acceptlicense yes \
    --quiet
```

#### Database Configuration

```objectscript
// Configure namespace for ML models
Set ns = "ML"
ZN ns

// Enable IntegratedML
Do $system.SQL.Execute("CREATE TABLE IF NOT EXISTS MLModels (ModelName VARCHAR(255), ModelData VARBINARY)")

// Configure resource allocation
Set ^%SYS("LOCKSIZ") = 33554432
Set ^%SYS("GMHEAP") = 268435456
Set ^%SYS("GLOSIZE") = 1073741824

// Configure journal settings for performance
Set ^%SYS("JOURNAL","CurrentSize") = 1024
Set ^%SYS("JOURNAL","MaxSize") = 8192
Set ^%SYS("JOURNAL","SyncCommit") = 0

// Enable SQL optimization
Set ^%SYS("SQL","DefaultSchema") = "ML"
Set ^%SYS("SQL","SqlMode") = 2
```

#### Performance Tuning

```ini
# /opt/intersystems/ML/iris.cpf
[ConfigFile]
Product=IRIS
Version=2024.1

[config]
globals=32G
routines=1G
locksiz=134217728
gmheap=2G

[Databases]
ML=/opt/intersystems/ML/mgr/ml/

[SystemMode]
SystemMode=LIVE

[Memory]
LockHashTableSize=131072
GlobalBufferCount=524288

[SQL]
DefaultSchema=ML
SqlMode=2
```

### Database Security Configuration

```objectscript
// Create dedicated ML user
Set user = "mlservice"
Set password = "SecureMLPassword123!"

// Create user with minimal privileges
Do ##class(%SYS.UserManager).CreateUser(user, password)

// Grant necessary permissions
Set $NAMESPACE = "ML"
Do ##class(%SQL.Manager.API).CreateRole("MLRole")
Do ##class(%SQL.Manager.API).GrantPrivilege("MLRole", "%All", "TABLE", "MLModels")
Do ##class(%SQL.Manager.API).GrantRole(user, "MLRole")

// Configure SSL/TLS
Set ssl = ##class(%SYS.SSLConfig).%New()
Set ssl.Name = "MLService"
Set ssl.CertificateFile = "/opt/certs/server.crt"
Set ssl.PrivateKeyFile = "/opt/certs/server.key"
Do ssl.%Save()
```

---

## 🤖 Model Deployment Strategies

### Blue-Green Deployment

Zero-downtime model updates using parallel environments.

```python
# deployment/blue_green.py
class BlueGreenDeployment:
    def __init__(self, config):
        self.config = config
        self.blue_env = ModelEnvironment("blue")
        self.green_env = ModelEnvironment("green")
        self.current_env = "blue"
    
    def deploy_new_version(self, model_path, version):
        """Deploy new model version with zero downtime."""
        inactive_env = "green" if self.current_env == "blue" else "blue"
        target_env = getattr(self, f"{inactive_env}_env")
        
        # Deploy to inactive environment
        target_env.deploy_model(model_path, version)
        
        # Run health checks
        if not target_env.health_check():
            raise DeploymentError("Health check failed")
        
        # Switch traffic
        self.switch_traffic(inactive_env)
        self.current_env = inactive_env
        
        # Clean up old environment
        old_env = getattr(self, f"{'green' if inactive_env == 'blue' else 'blue'}_env")
        old_env.cleanup()
    
    def switch_traffic(self, target_env):
        """Switch load balancer to target environment."""
        lb_config = {
            "upstream": f"ml_models_{target_env}",
            "servers": self.config[f"{target_env}_servers"]
        }
        self.update_load_balancer(lb_config)
```

### Canary Deployment

Gradual rollout with traffic splitting for risk mitigation.

```python
# deployment/canary.py
class CanaryDeployment:
    def __init__(self, config):
        self.config = config
        self.traffic_split = 0  # Start with 0% canary traffic
    
    def start_canary(self, model_path, version):
        """Start canary deployment with 5% traffic."""
        self.deploy_canary_version(model_path, version)
        self.set_traffic_split(5)
        self.monitor_metrics()
    
    def promote_canary(self):
        """Gradually increase canary traffic."""
        stages = [10, 25, 50, 75, 100]
        for split in stages:
            self.set_traffic_split(split)
            time.sleep(self.config.get('stage_duration', 300))
            
            if not self.metrics_healthy():
                self.rollback_canary()
                raise DeploymentError("Metrics degraded during canary")
        
        self.finalize_deployment()
    
    def metrics_healthy(self):
        """Check if canary metrics are within acceptable bounds."""
        metrics = self.get_canary_metrics()
        return (
            metrics['error_rate'] < self.config['max_error_rate'] and
            metrics['latency_p95'] < self.config['max_latency'] and
            metrics['accuracy'] > self.config['min_accuracy']
        )
```

### A/B Testing Deployment

Statistical testing for model performance comparison.

```python
# deployment/ab_testing.py
class ABTestingDeployment:
    def __init__(self, config):
        self.config = config
        self.variant_a = None
        self.variant_b = None
        self.test_results = {}
    
    def setup_ab_test(self, model_a, model_b, traffic_split=50):
        """Setup A/B test with two model versions."""
        self.variant_a = self.deploy_model(model_a, "variant_a")
        self.variant_b = self.deploy_model(model_b, "variant_b")
        
        # Configure traffic routing
        routing_config = {
            "variant_a": traffic_split,
            "variant_b": 100 - traffic_split
        }
        self.configure_routing(routing_config)
    
    def collect_results(self, duration_hours=24):
        """Collect A/B test results over specified duration."""
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        while time.time() < end_time:
            self.update_test_metrics()
            time.sleep(self.config.get('collection_interval', 300))
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Statistical analysis of A/B test results."""
        from scipy import stats
        
        metrics_a = self.get_variant_metrics("variant_a")
        metrics_b = self.get_variant_metrics("variant_b")
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(
            metrics_a['accuracy_scores'],
            metrics_b['accuracy_scores']
        )
        
        return {
            'winner': 'variant_b' if metrics_b['mean_accuracy'] > metrics_a['mean_accuracy'] else 'variant_a',
            'confidence': 1 - p_value,
            'effect_size': abs(metrics_b['mean_accuracy'] - metrics_a['mean_accuracy']),
            'statistical_significance': p_value < 0.05
        }
```

---

## 🐳 Container Deployment

### Docker Configuration

#### Base Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash mlservice
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mlservice:mlservice . .

# Set up model directory
RUN mkdir -p /app/models && chown mlservice:mlservice /app/models

# Switch to non-root user
USER mlservice

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

#### Multi-stage Production Dockerfile

```dockerfile
# Dockerfile.production
# Build stage
FROM python:3.9-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim-bullseye AS production

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/home/mlservice/.local/bin:$PATH

RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    libssl1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash mlservice

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/mlservice/.local

# Copy application
COPY --chown=mlservice:mlservice . .

USER mlservice

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

### Docker Compose Configuration

#### Development Environment

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  iris:
    image: intersystemsdc/iris-community:2024.1
    environment:
      - ISC_PASSWORD=YourSecurePassword123!
    ports:
      - "1972:1972"
      - "52773:52773"
    volumes:
      - iris_data:/dur/irisdata
      - ./config/iris:/opt/irisapp
    networks:
      - ml_network

  ml_service:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=iris://mlservice:password@iris:1972/ML
      - LOG_LEVEL=DEBUG
      - ENVIRONMENT=development
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - iris
    networks:
      - ml_network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ml_network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ml_network

volumes:
  iris_data:
  grafana_data:

networks:
  ml_network:
    driver: bridge
```

#### Production Environment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  iris_primary:
    image: intersystemsdc/iris-community:2024.1
    environment:
      - ISC_PASSWORD_FILE=/run/secrets/iris_password
    ports:
      - "1972:1972"
    volumes:
      - iris_primary_data:/dur/irisdata
      - ./config/iris:/opt/irisapp
    secrets:
      - iris_password
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 8G
          cpus: '4'
    networks:
      - ml_network

  iris_mirror:
    image: intersystemsdc/iris-community:2024.1
    environment:
      - ISC_PASSWORD_FILE=/run/secrets/iris_password
      - MIRROR_MODE=async
    volumes:
      - iris_mirror_data:/dur/irisdata
    secrets:
      - iris_password
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 8G
          cpus: '4'
    networks:
      - ml_network

  ml_service:
    build:
      context: .
      dockerfile: Dockerfile.production
    environment:
      - DATABASE_URL_FILE=/run/secrets/database_url
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    secrets:
      - database_url
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    networks:
      - ml_network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - ml_service
    deploy:
      replicas: 2
    networks:
      - ml_network

secrets:
  iris_password:
    external: true
  database_url:
    external: true

volumes:
  iris_primary_data:
  iris_mirror_data:

networks:
  ml_network:
    driver: overlay
    attachable: true
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: integratedml
  labels:
    name: integratedml

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: integratedml
data:
  app.yaml: |
    database:
      host: iris-service
      port: 1972
      namespace: ML
    
    models:
      cache_size: "2GB"
      workers: 4
      batch_size: 1000
    
    logging:
      level: INFO
      format: json
    
    monitoring:
      metrics_port: 9090
      health_check_path: /health
```

#### Deployment Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: integratedml
  labels:
    app: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: integratedml/ml-service:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: ml-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: integratedml
spec:
  selector:
    app: ml-service
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-ingress
  namespace: integratedml
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ml.yourdomain.com
    secretName: ml-tls
  rules:
  - host: ml.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-service
            port:
              number: 80
```

#### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: integratedml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### ECS Fargate Configuration

```yaml
# aws/ecs-task-definition.json
{
  "family": "integratedml-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ml-service",
      "image": "account.dkr.ecr.region.amazonaws.com/integratedml:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/integratedml",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### CloudFormation Template

```yaml
# aws/cloudformation.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'IntegratedML Production Infrastructure'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID for deployment
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs for ECS service
  CertificateArn:
    Type: String
    Description: SSL certificate ARN

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: integratedml-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1
        - CapacityProvider: FARGATE_SPOT
          Weight: 4

  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: integratedml-alb
      Type: application
      Scheme: internet-facing
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Subnets: !Ref SubnetIds

  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: integratedml-targets
      Port: 8000
      Protocol: HTTP
      TargetType: ip
      VpcId: !Ref VpcId
      HealthCheckPath: /health
      HealthCheckProtocol: HTTP
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 5

  ECSService:
    Type: AWS::ECS::Service
    DependsOn: HTTPSListener
    Properties:
      ServiceName: integratedml-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      LaunchType: FARGATE
      DesiredCount: 3
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref ECSSecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: ml-service
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup

  AutoScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 20
      MinCapacity: 3
      ResourceId: !Sub service/${ECSCluster}/${ECSService.Name}
      RoleARN: !Sub arn:aws:iam::${AWS::AccountId}:role/aws-service-role/ecs.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_ECSService
      ScalableDimension: ecs:service:DesiredCount
      ServiceNamespace: ecs

  AutoScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: integratedml-scaling-policy
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref AutoScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ECSServiceAverageCPUUtilization
        TargetValue: 70
        ScaleOutCooldown: 300
        ScaleInCooldown: 300
```

### Azure Deployment

#### Container Instances Configuration

```yaml
# azure/container-group.yaml
apiVersion: 2019-12-01
location: East US
name: integratedml-container-group
properties:
  containers:
  - name: ml-service
    properties:
      image: acrregistry.azurecr.io/integratedml:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
        limits:
          cpu: 4
          memoryInGb: 8
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: iris://user:pass@host:1972/ML
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: integratedml-ml-service
tags:
  Environment: Production
  Application: IntegratedML
```

### Google Cloud Platform Deployment

#### Cloud Run Configuration

```yaml
# gcp/cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: integratedml-service
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/client-name: gcloud
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "3"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/integratedml:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          timeoutSeconds: 5
          periodSeconds: 10
          successThreshold: 1
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          timeoutSeconds: 5
          periodSeconds: 30
```

---

## ⚡ Performance Optimization

### Application-Level Optimization

#### Model Caching Strategy

```python
# optimization/model_cache.py
import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CacheEntry:
    model: Any
    timestamp: float
    access_count: int
    last_access: float

class IntelligentModelCache:
    def __init__(self, max_size: int = 10, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self._start_cleanup_thread()
    
    def get_model(self, model_key: str, loader_func: callable) -> Any:
        """Get model with intelligent caching and loading."""
        with self.lock:
            current_time = time.time()
            
            # Check if model exists and is valid
            if model_key in self.cache:
                entry = self.cache[model_key]
                if current_time - entry.timestamp < self.ttl:
                    entry.access_count += 1
                    entry.last_access = current_time
                    return entry.model
                else:
                    del self.cache[model_key]
            
            # Load model if not cached or expired
            model = loader_func()
            
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Cache the model
            self.cache[model_key] = CacheEntry(
                model=model,
                timestamp=current_time,
                access_count=1,
                last_access=current_time
            )
            
            return model
    
    def _evict_lru(self):
        """Evict least recently used model."""
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_access
        )
        del self.cache[lru_key]
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup."""
        def cleanup():
            while True:
                time.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_expired()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time - entry.timestamp > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
```

#### Batch Prediction Optimization

```python
# optimization/batch_predictor.py
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class BatchPredictor:
    def __init__(self, model_cache, max_batch_size: int = 1000, max_workers: int = 4):
        self.model_cache = model_cache
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def predict_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch predictions with optimal batching and parallelization."""
        # Group requests by model
        model_groups = self._group_by_model(requests)
        
        # Process each model group
        tasks = []
        for model_key, group_requests in model_groups.items():
            task = self._process_model_group(model_key, group_requests)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten and reorder results
        return self._merge_results(results, requests)
    
    def _group_by_model(self, requests: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group requests by model type for batch processing."""
        groups = {}
        for req in requests:
            model_key = req['model_key']
            if model_key not in groups:
                groups[model_key] = []
            groups[model_key].append(req)
        return groups
    
    async def _process_model_group(self, model_key: str, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all requests for a specific model."""
        model = self.model_cache.get_model(model_key, lambda: self._load_model(model_key))
        
        # Split into optimal batch sizes
        batches = self._create_batches(requests, self.max_batch_size)
        
        # Process batches in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self._predict_batch_sync,
                model,
                batch
            )
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten batch results
        return [result for batch in batch_results for result in batch]
    
    def _predict_batch_sync(self, model: Any, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous batch prediction for thread executor."""
        # Prepare batch input
        X_batch = np.array([req['features'] for req in batch])
        
        # Make predictions
        predictions = model.predict(X_batch)
        
        # Format results
        results = []
        for i, req in enumerate(batch):
            results.append({
                'request_id': req['request_id'],
                'prediction': predictions[i],
                'model_key': req['model_key']
            })
        
        return results
```

### Database Optimization

#### Connection Pooling

```python
# optimization/db_pool.py
import queue
import threading
import time
from contextlib import contextmanager
from typing import Optional

class DatabaseConnectionPool:
    def __init__(self, 
                 connection_factory: callable,
                 min_connections: int = 5,
                 max_connections: int = 20,
                 max_idle_time: int = 300):
        self.connection_factory = connection_factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self.pool = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # Initialize minimum connections
        self._initialize_pool()
        
        # Start maintenance thread
        self._start_maintenance_thread()
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections."""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self):
        """Create new database connection with metadata."""
        with self.lock:
            if self.active_connections >= self.max_connections:
                raise Exception("Maximum connections exceeded")
            
            conn = self.connection_factory()
            self.active_connections += 1
            
            return {
                'connection': conn,
                'created_at': time.time(),
                'last_used': time.time()
            }
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager."""
        conn_info = None
        try:
            # Try to get connection from pool
            try:
                conn_info = self.pool.get_nowait()
            except queue.Empty:
                # Create new connection if pool is empty
                conn_info = self._create_connection()
            
            # Update last used time
            conn_info['last_used'] = time.time()
            
            # Yield connection
            yield conn_info['connection']
            
        finally:
            # Return connection to pool
            if conn_info:
                try:
                    self.pool.put_nowait(conn_info)
                except queue.Full:
                    # Pool is full, close connection
                    self._close_connection(conn_info)
    
    def _close_connection(self, conn_info):
        """Close connection and update count."""
        try:
            conn_info['connection'].close()
        except:
            pass
        
        with self.lock:
            self.active_connections -= 1
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread."""
        def maintenance():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_connections()
        
        thread = threading.Thread(target=maintenance, daemon=True)
        thread.start()
    
    def _cleanup_idle_connections(self):
        """Remove idle connections beyond minimum."""
        current_time = time.time()
        connections_to_close = []
        connections_to_keep = []
        
        # Drain pool
        while True:
            try:
                conn_info = self.pool.get_nowait()
                if (current_time - conn_info['last_used'] > self.max_idle_time and
                    len(connections_to_keep) >= self.min_connections):
                    connections_to_close.append(conn_info)
                else:
                    connections_to_keep.append(conn_info)
            except queue.Empty:
                break
        
        # Close idle connections
        for conn_info in connections_to_close:
            self._close_connection(conn_info)
        
        # Return active connections to pool
        for conn_info in connections_to_keep:
            self.pool.put(conn_info)
```

### Load Balancing Configuration

#### NGINX Configuration

```nginx
# /etc/nginx/sites-available/integratedml
upstream ml_backend {
    least_conn;
    server 127.0.0.1:8001 weight=3 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 weight=3 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 weight=3 max_fails=3 fail_timeout=30s;
    
    # Backup server
    server 127.0.0.1:8004 backup;
    
    # Health check
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name ml.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/ml.yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/ml.yourdomain.com.key;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubdomains";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req zone=api burst=20 nodelay;
    
    # Logging
    access_log /var/log/nginx/ml_access.log combined;
    error_log /var/log/nginx/ml_error.log;
    
    location / {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_timeout 30s;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        access_log off;
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
    }
    
    location /metrics {
        access_log off;
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://ml_backend;
    }
}
```

This comprehensive deployment documentation provides production-ready strategies for deploying IntegratedML Flexible Model Integration across various environments, from single servers to large-scale cloud deployments with complete performance optimization, security, and monitoring capabilities.