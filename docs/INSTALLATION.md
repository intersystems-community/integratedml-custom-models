# IntegratedML Custom Models - Installation Guide

**Program Status**: Early Access Program (EAP)
**Last Updated**: 2025-01-12
**Target**: IRIS 2026.1

---

## Purpose

This guide provides comprehensive installation instructions for IntegratedML Custom Models. Follow the platform-specific instructions for your operating system.

**Target**: Complete installation in <30 minutes

**Platform Priority**:
- **macOS**: Primary support (most tested)
- **Linux**: Secondary support (Ubuntu 22.04+)
- **Windows**: Secondary support (WSL2 or Docker recommended)

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [macOS Installation (Primary Platform)](#macos-installation-primary-platform)
- [Linux Installation (Secondary Platform)](#linux-installation-secondary-platform)
- [Windows Installation (Secondary Platform)](#windows-installation-secondary-platform)
- [Installation Verification](#installation-verification)
- [Post-Installation Setup](#post-installation-setup)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | macOS 13+, Ubuntu 22.04+, Windows 11 | Windows requires WSL2 or Docker |
| **IRIS Version** | 2025.2+ | Community Edition or licensed |
| **Python** | 3.8+ | 3.11+ recommended for AutoML compatibility |
| **Memory** | 8GB RAM | 16GB recommended for large models |
| **Disk Space** | 5GB free | 10GB recommended for all demos |
| **Docker** | Latest stable | Docker Desktop (macOS/Windows) or Docker Engine (Linux) |

### Supported Platform Matrix

| Platform | Docker | Local Install | Support Level | Tested Versions |
|----------|--------|---------------|---------------|-----------------|
| **macOS** | ✅ Recommended | ✅ Supported | Primary | 13 (Ventura), 14 (Sonoma) |
| **Linux (Ubuntu)** | ✅ Recommended | ✅ Supported | Secondary | 22.04 LTS, 24.04 LTS |
| **Linux (Other)** | ✅ Recommended | ⚠️ Untested | Community | Debian, Fedora, Arch |
| **Windows** | ✅ Recommended | ⚠️ Use WSL2 | Secondary | Windows 11 + Docker Desktop |

**Legend**:
- ✅ Fully supported and tested
- ⚠️ Limited testing, may require troubleshooting
- ❌ Not supported

### Python Version Compatibility

| Python Version | Custom Models | AutoML | Recommended |
|----------------|---------------|--------|-------------|
| 3.11+ | ✅ | ✅ | ✅ **Best choice** |
| 3.10 | ✅ | ✅ | ✅ Fully supported |
| 3.9 | ✅ | ⚠️ | ⚠️ AutoML may have issues |
| 3.8 | ✅ | ⚠️ | ⚠️ Minimum version |
| <3.8 | ❌ | ❌ | ❌ Not supported |

---

## Installation Methods

### Method 1: Docker (Recommended)

**Best for**: Quick setup, cross-platform consistency, EAP evaluation

**Pros**:
- ✅ Fastest setup (~15-20 minutes)
- ✅ Isolated environment
- ✅ Consistent across platforms
- ✅ Includes all dependencies
- ✅ Easy cleanup

**Cons**:
- ⚠️ Requires Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- ⚠️ Higher resource usage (8GB RAM minimum)

**Use when**: First-time setup, EAP evaluation, learning Custom Models

---

### Method 2: Local Installation

**Best for**: Development, production evaluation, existing IRIS installations

**Pros**:
- ✅ Better performance
- ✅ Works with existing IRIS installations
- ✅ More control over configuration
- ✅ Lower resource usage

**Cons**:
- ⚠️ More complex setup (~25-30 minutes)
- ⚠️ Platform-specific dependencies
- ⚠️ Requires manual IRIS configuration

**Use when**: Integrating with existing IRIS, production evaluation, development

---

## macOS Installation (Primary Platform)

### Prerequisites

1. **macOS 13 (Ventura) or later**
2. **Homebrew** (package manager):
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. **Git**:
   ```bash
   brew install git
   ```
4. **Python 3.11+** (recommended):
   ```bash
   brew install python@3.11
   ```

### Method 1: Docker Installation (macOS)

#### Step 1: Install Docker Desktop

```bash
# Download Docker Desktop for Mac from https://www.docker.com/products/docker-desktop
# Or install via Homebrew
brew install --cask docker

# Start Docker Desktop from Applications
open -a Docker
```

**Verify Docker**:
```bash
docker --version
docker-compose --version
```

**Configure Docker Desktop**:
- Open Docker Desktop → Preferences → Resources
- Set Memory to at least 8GB (12GB recommended)
- Set CPU to at least 4 cores
- Apply & Restart

---

#### Step 2: Clone Repository

```bash
# Clone repository
git clone https://github.com/intersystems-community/integratedml-custom-models.git
cd integratedml-custom-models

# Verify you're in the correct directory
ls -la
# Should see: README.md, docker-compose.yml, Makefile, etc.
```

---

#### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env (optional - defaults work for most users)
nano .env  # or use your preferred editor
```

**Key Configuration Options** (optional):
```bash
# Database credentials (change for production)
IRIS_USERNAME=demo
IRIS_PASSWORD=demo
IRIS_NAMESPACE=USER

# Ports (change if conflicts)
IRIS_PORT=1972
IRIS_WEB_PORT=52773

# Demo data sizes
CREDIT_RISK_SAMPLES=10000
FRAUD_DETECTION_SAMPLES=25000
SALES_FORECASTING_DAYS=365
```

---

#### Step 4: Install and Start IRIS

```bash
# One-command setup (recommended for first-time users)
make setup

# This will:
# 1. Install Python dependencies
# 2. Start IRIS container
# 3. Wait for IRIS to be ready
# 4. Verify installation
```

**Expected output**:
```
✅ Installing Python dependencies...
✅ Starting IRIS database...
✅ Waiting for IRIS to be ready...
✅ IRIS is running and healthy
✅ Installation complete!
```

**Time**: ~15-20 minutes (includes Docker image download)

---

#### Step 5: Verify Installation

```bash
# Run verification script
make test-connection

# Expected output:
# ✅ IRIS connection successful
# ✅ IntegratedML available
# ✅ Custom Models enabled
# ✅ All systems ready
```

**Manual verification**:
```bash
# Check Docker containers
docker ps

# You should see:
# - integratedml-custom-models-iris (IRIS database)

# Access IRIS Management Portal
open http://localhost:52773/csp/sys/UtilHome.csp
# Username: demo
# Password: demo
```

---

#### Step 6: Run First Demo

```bash
# Run Credit Risk demo to verify everything works
make demo-credit

# Expected output:
# ✅ Generating test data...
# ✅ Creating model...
# ✅ Training model...
# ✅ Making predictions...
# ✅ Model accuracy: ~85%
# ✅ Demo completed successfully
```

**Time**: ~2-3 minutes

---

### Method 2: Local Installation (macOS)

**Use this if**: You want better performance, have existing IRIS installation, or need more control

#### Step 1: Install IRIS

**Option A: Download IRIS Community Edition**

1. Visit: https://www.intersystems.com/try-intersystems-iris-for-free/
2. Download IRIS Community Edition 2025.2+ for macOS
3. Run installer: `IRIS-2025.2.0.xxx.dmg`
4. Follow installation wizard (default settings work)

**Default installation path**: `/usr/local/bin/iris`

**Option B: Use Existing IRIS Installation**

If you have IRIS 2025.2+ already installed:
```bash
# Verify IRIS version
iris version

# Should show: InterSystems IRIS Version 2025.2.0 or later
```

---

#### Step 2: Install Python and Dependencies

```bash
# Install Python 3.11 (if not already installed)
brew install python@3.11

# Verify Python version
python3.11 --version

# Install uv (fast Python package installer) - recommended
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
python3.11 -m pip install --upgrade pip
```

---

#### Step 3: Clone Repository and Install

```bash
# Clone repository
git clone https://github.com/intersystems-community/integratedml-custom-models.git
cd integratedml-custom-models

# Install dependencies with uv (recommended)
make install

# Or install with pip
pip install -r requirements.txt
pip install -e .
```

---

#### Step 4: Configure IRIS Connection

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your IRIS connection details
nano .env
```

**Update these values**:
```bash
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM  # or your IRIS username
IRIS_PASSWORD=<your-password>
```

---

#### Step 5: Install IntegratedML Python Package in IRIS

This step installs the IntegratedML AutoML package into IRIS's Python environment:

```bash
# Connect to IRIS terminal
iris session IRIS

# In IRIS terminal:
USER> set sc = ##class(%SYSTEM.Python).Install("/path/to/python3.11")
USER> do ##class(%SYSTEM.Python).GetInfo()

# Verify Python is configured
USER> write ##class(%SYSTEM.Python).GetPythonPath()

# Install IntegratedML
USER> do ##class(%SYSTEM.Python).Shell("pip install --index-url https://registry.intersystems.com/pypi/simple --no-cache-dir --target /usr/irissys/mgr/python intersystems-iris-automl")

# Create symlink (required for IntegratedML to work)
USER> do ##class(%SYSTEM.Python).Shell("ln -sf /usr/irissys/mgr/python/iris_automl /opt/irisapp/data/mgr/python/iris_automl")

# Exit IRIS terminal
USER> halt
```

---

#### Step 6: Deploy Custom Models to IRIS

```bash
# Create custom models directory in IRIS
mkdir -p /opt/irisapp/data/mgr/python/custom_models/classifiers
mkdir -p /opt/irisapp/data/mgr/python/custom_models/regressors

# Copy demo models to IRIS
cp demos/credit_risk/models/credit_risk_classifier.py /opt/irisapp/data/mgr/python/custom_models/classifiers/
cp demos/fraud_detection/models/ensemble_fraud_detector.py /opt/irisapp/data/mgr/python/custom_models/classifiers/
cp demos/sales_forecasting/models/hybrid_forecasting_model.py /opt/irisapp/data/mgr/python/custom_models/regressors/
cp demos/dna_similarity/models/dna_classifier.py /opt/irisapp/data/mgr/python/custom_models/classifiers/

# Restart IRIS to load models
iris stop IRIS quietly
iris start IRIS
```

---

#### Step 7: Verify Installation

```bash
# Test database connection
python -c "from shared.database import test_connection; print('✅ Success' if test_connection() else '❌ Failed')"

# Run Credit Risk demo
python demos/credit_risk/main.py

# Expected output:
# ✅ Generating test data...
# ✅ Creating model...
# ✅ Training model...
# ✅ Making predictions...
# ✅ Model accuracy: ~85%
```

---

## Linux Installation (Secondary Platform)

### Prerequisites (Ubuntu 22.04+)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install prerequisites
sudo apt install -y git curl build-essential libssl-dev

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

### Method 1: Docker Installation (Linux)

```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt install -y docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

**Continue with Steps 2-6 from [macOS Docker Installation](#method-1-docker-installation-macos)**

**Linux-Specific Notes**:

1. **Volume Permissions**: May need to fix permissions:
   ```bash
   sudo chown -R 51773:51773 ./data
   ```

2. **Firewall**: Ensure ports 1972 and 52773 are open:
   ```bash
   sudo ufw allow 1972
   sudo ufw allow 52773
   ```

### Method 2: Local Installation (Linux)

Follow [macOS Local Installation](#method-2-local-installation-macos) with these adjustments:

**IRIS Installation**:
- Download IRIS for Linux from InterSystems
- Install with: `sudo dpkg -i IRIS-2025.2.0.xxx.deb` (Debian/Ubuntu)
- Or: `sudo rpm -i IRIS-2025.2.0.xxx.rpm` (Red Hat/CentOS)

**Python Setup**:
```bash
# Use system Python or install specific version
sudo apt install -y python3.11 python3.11-venv python3-pip
```

---

## Windows Installation (Secondary Platform)

### Prerequisites

**Recommended**: Windows 11 with WSL2 or Docker Desktop

### Method 1: Docker Desktop (Windows - Recommended)

#### Step 1: Install WSL2

```powershell
# Run in PowerShell as Administrator
wsl --install

# Restart computer

# Verify WSL2
wsl --version
```

#### Step 2: Install Docker Desktop

1. Download Docker Desktop for Windows from https://www.docker.com/products/docker-desktop
2. Install Docker Desktop
3. Enable WSL2 backend in Docker Desktop settings
4. Restart Docker Desktop

#### Step 3: Install Git

Download and install Git for Windows: https://git-scm.com/download/win

#### Step 4: Follow Installation in WSL2

```bash
# Open WSL2 terminal (Ubuntu)
wsl

# Follow Linux installation steps
# (See "Linux Installation" section above)
```

**Windows-Specific Notes**:

1. **Line Endings**: Configure Git to handle line endings:
   ```bash
   git config --global core.autocrlf input
   ```

2. **File Permissions**: Windows file permissions may cause issues. Use Docker volumes instead:
   ```bash
   docker volume create iml-custom-models-data
   ```

3. **Performance**: WSL2 file I/O is slower for files in Windows filesystem. Keep project in WSL2 filesystem:
   ```bash
   # Good: /home/user/integratedml-custom-models
   # Bad: /mnt/c/Users/user/integratedml-custom-models
   ```

---

### Method 2: Local Installation (Windows - Advanced)

**Not recommended for EAP**. Use WSL2 + Docker instead.

If you must use native Windows:
1. Install Python 3.11+ from python.org
2. Install IRIS for Windows
3. Use PowerShell or CMD with admin privileges
4. Follow macOS local installation steps, adapting paths

---

## Installation Verification

### Automated Verification

```bash
# Run comprehensive verification
make verify

# Expected output:
# ✅ Docker is running
# ✅ IRIS is healthy
# ✅ Database connection successful
# ✅ IntegratedML enabled
# ✅ Custom Models available
# ✅ Python dependencies installed
# ✅ Demo models accessible
# ✅ All systems ready
```

### Manual Verification Checklist

- [ ] **Docker running** (if using Docker method)
  ```bash
  docker ps
  # Should show IRIS container
  ```

- [ ] **IRIS accessible**
  - Open: http://localhost:52773/csp/sys/UtilHome.csp
  - Login with configured credentials

- [ ] **Database connection works**
  ```bash
  python -c "from shared.database import test_connection; test_connection()"
  ```

- [ ] **IntegratedML enabled**
  ```sql
  -- In IRIS SQL:
  SELECT * FROM INFORMATION_SCHEMA.ML_MODELS
  ```

- [ ] **Custom models accessible**
  ```bash
  ls /opt/irisapp/data/mgr/python/custom_models/classifiers/
  # Should show model files
  ```

- [ ] **Demo runs successfully**
  ```bash
  make demo-credit
  # Should complete without errors
  ```

---

## Post-Installation Setup

### 1. Configure Development Environment

```bash
# Set up your preferred Python environment
python3.11 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install development tools
pip install black flake8 mypy pytest
```

### 2. Explore Documentation

**Recommended reading order**:
1. [`QUICK_GUIDE_CUSTOM_MODELS.md`](QUICK_GUIDE_CUSTOM_MODELS.md) - 5-minute overview
2. [`user_guide.md`](user_guide.md) - Step-by-step usage
3. [`architecture.md`](architecture.md) - System design
4. [`api_reference.md`](api_reference.md) - API documentation

### 3. Run All Demos

```bash
# Run all demos to verify full functionality
make demos

# Or run individually:
make demo-credit
make demo-fraud
make demo-sales
make demo-dna
```

### 4. Complete EAP Entry Survey

**If you're an EAP participant**:
- Check your email for entry survey link
- Complete survey with installation feedback
- Report any issues encountered during setup

---

## Troubleshooting

For detailed troubleshooting, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

### Quick Fixes

#### Docker Issues

**Issue**: "Cannot connect to Docker daemon"
```bash
# Start Docker Desktop (macOS/Windows)
open -a Docker

# Or start Docker service (Linux)
sudo systemctl start docker
```

**Issue**: "Port already in use"
```bash
# Find process using port
lsof -i :1972
lsof -i :52773

# Kill process or change port in .env
```

**Issue**: "Docker container won't start"
```bash
# Check logs
docker logs integratedml-custom-models-iris

# Remove and recreate
make clean
make setup
```

#### IRIS Issues

**Issue**: "IntegratedML symlink missing"
```bash
# Create symlink manually
docker exec -it integratedml-custom-models-iris bash
ln -sf /usr/irissys/mgr/python/iris_automl /opt/irisapp/data/mgr/python/iris_automl
exit
docker restart integratedml-custom-models-iris
```

**Issue**: "Model not found"
```bash
# Verify model file exists
docker exec -it integratedml-custom-models-iris ls /opt/irisapp/data/mgr/python/custom_models/classifiers/

# Copy model if missing
docker cp demos/credit_risk/models/credit_risk_classifier.py integratedml-custom-models-iris:/opt/irisapp/data/mgr/python/custom_models/classifiers/

# Restart IRIS
docker restart integratedml-custom-models-iris
```

#### Python Issues

**Issue**: "Module not found: iris_automl"
```bash
# Reinstall IntegratedML package
docker exec -it integratedml-custom-models-iris bash
python -m pip install --index-url https://registry.intersystems.com/pypi/simple --no-cache-dir --target /usr/irissys/mgr/python intersystems-iris-automl
exit
docker restart integratedml-custom-models-iris
```

**Issue**: "Import error: No module named 'shared'"
```bash
# Install project in editable mode
pip install -e .
```

### Platform-Specific Issues

**macOS**: Rosetta 2 required for M1/M2 Macs
```bash
# Install Rosetta 2 if needed
softwareupdate --install-rosetta --agree-to-license
```

**Linux**: Permission denied errors
```bash
# Fix Docker permissions
sudo chown -R 51773:51773 ./data
```

**Windows**: Slow performance in WSL2
```bash
# Use WSL2 filesystem (not /mnt/c/)
cd ~
git clone https://github.com/intersystems-community/integratedml-custom-models.git
```

---

## Getting Help

**If installation takes >30 minutes or you encounter issues**:

1. **Check Known Issues**: [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md)
2. **Check Troubleshooting**: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
3. **Check FAQ**: [`EAP_FAQ.md`](EAP_FAQ.md)
4. **Contact Support**:
   - Email: thomas.dyar@intersystems.com
   - Include: OS, Python version, error messages, installation method

---

## Next Steps

After successful installation:

1. **Run a demo**: `make demo-credit`
2. **Read the user guide**: [`user_guide.md`](user_guide.md)
3. **Review architecture**: [`architecture.md`](architecture.md)
4. **Complete EAP entry survey**: Check your email for link
5. **Explore your use case**: Identify how Custom Models could help you

---

**Congratulations! You're ready to use IntegratedML Custom Models!**

**— The InterSystems Data Platforms Product Team**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-12
**Platform Focus**: macOS (primary), Linux/Windows (secondary)
