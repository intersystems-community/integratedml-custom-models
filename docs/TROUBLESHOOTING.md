# IntegratedML Custom Models - Troubleshooting Guide

**Program Status**: Early Access Program (EAP)
**Last Updated**: 2025-01-12
**Target**: IRIS 2026.1

---

## Purpose

This guide helps you diagnose and resolve common issues with IntegratedML Custom Models. Issues are organized by category with step-by-step solutions.

**Before contacting support**:
1. ✅ Check this troubleshooting guide
2. ✅ Check [`EAP_KNOWN_ISSUES.md`](EAP_KNOWN_ISSUES.md) for known limitations
3. ✅ Check [`EAP_FAQ.md`](EAP_FAQ.md) for frequently asked questions

**If still stuck**: Email thomas.dyar@intersystems.com with detailed error information.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [IRIS Connection Issues](#iris-connection-issues)
- [Model Loading Issues](#model-loading-issues)
- [Training Issues](#training-issues)
- [Prediction Issues](#prediction-issues)
- [Performance Issues](#performance-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Demo-Specific Issues](#demo-specific-issues)
- [Diagnostic Information Collection](#diagnostic-information-collection)

---

## Installation Issues

### Issue: Installation takes longer than 30 minutes

**Symptoms**:
- Docker image download is very slow
- `make setup` hangs

**Causes**:
- Slow internet connection
- Docker downloading large IRIS image (~2GB)

**Solutions**:

1. **Check internet connection**:
   ```bash
   # Test download speed
   curl -o /dev/null http://speedtest.wdc01.softlayer.com/downloads/test10.zip
   ```

2. **Download Docker image separately** (allows monitoring progress):
   ```bash
   # Pull IRIS image manually
   docker pull intersystemsdc/iris-community:latest

   # Then run setup
   make setup
   ```

3. **Use local IRIS installation** instead of Docker (if available):
   - See [`INSTALLATION.md`](INSTALLATION.md) → Method 2: Local Installation

---

### Issue: "Permission denied" during installation

**Symptoms**:
- `make setup` fails with permission errors
- Cannot create directories or files

**Causes**:
- Insufficient permissions
- Docker not running as current user (Linux)

**Solutions**:

**macOS/Windows**:
```bash
# Ensure Docker Desktop is running
open -a Docker  # macOS

# Try with sudo (not recommended long-term)
sudo make setup
```

**Linux**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix directory permissions
sudo chown -R $(whoami):$(whoami) .

# Retry
make setup
```

---

### Issue: Python version not supported

**Symptoms**:
- Error: "Python 3.8 or higher required"
- Import errors for modern Python features

**Causes**:
- Old Python version installed (3.6, 3.7)

**Solutions**:

**macOS**:
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Verify version
python3.11 --version

# Use specific Python version
python3.11 -m pip install -r requirements.txt
```

**Linux (Ubuntu)**:
```bash
# Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip

# Set as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

**Windows**:
- Download Python 3.11+ from python.org
- Ensure "Add to PATH" is checked during installation

---

### Issue: `make` command not found

**Symptoms**:
- `make setup` returns "command not found"

**Causes**:
- Make not installed (uncommon on macOS/Linux, common on Windows)

**Solutions**:

**macOS**:
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Linux**:
```bash
# Install build-essential
sudo apt install -y build-essential
```

**Windows**:
```bash
# Use WSL2 (recommended)
wsl

# Or install Make for Windows
choco install make  # Requires Chocolatey

# Or run commands manually without make
docker-compose up --build -d
pip install -r requirements.txt
```

---

## Docker Issues

### Issue: "Cannot connect to Docker daemon"

**Symptoms**:
- `docker ps` fails
- Error: "Is the docker daemon running?"

**Causes**:
- Docker Desktop not started (macOS/Windows)
- Docker service not running (Linux)

**Solutions**:

**macOS**:
```bash
# Start Docker Desktop
open -a Docker

# Wait for Docker to start (~30 seconds)
sleep 30

# Verify
docker ps
```

**Linux**:
```bash
# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Verify
docker ps
```

**Windows**:
- Open Docker Desktop from Start menu
- Wait for "Docker Desktop is running" notification

---

### Issue: "Port already in use"

**Symptoms**:
- Error: "Bind for 0.0.0.0:1972 failed: port is already allocated"
- Container won't start

**Causes**:
- Another IRIS instance running
- Another application using ports 1972 or 52773

**Solutions**:

1. **Find process using port**:
   ```bash
   # macOS/Linux
   lsof -i :1972
   lsof -i :52773

   # Windows (PowerShell)
   netstat -ano | findstr :1972
   ```

2. **Kill conflicting process**:
   ```bash
   # macOS/Linux (replace PID with actual process ID)
   kill -9 <PID>

   # Windows (PowerShell)
   taskkill /PID <PID> /F
   ```

3. **Change ports in .env**:
   ```bash
   # Edit .env file
   IRIS_PORT=1973
   IRIS_WEB_PORT=52774

   # Restart containers
   make clean
   make setup
   ```

---

### Issue: Docker container exits immediately

**Symptoms**:
- `docker ps` shows no containers
- `docker ps -a` shows container with "Exited" status

**Causes**:
- IRIS startup failure
- Insufficient Docker resources

**Solutions**:

1. **Check container logs**:
   ```bash
   docker logs integratedml-custom-models-iris

   # Look for errors like:
   # - "Out of memory"
   # - "Permission denied"
   # - "License error"
   ```

2. **Increase Docker resources**:
   - Open Docker Desktop → Preferences → Resources
   - Set Memory to 8GB minimum (12GB recommended)
   - Set CPU to 4 cores minimum
   - Apply & Restart

3. **Remove and recreate container**:
   ```bash
   make clean
   docker system prune -f  # Clean up Docker system
   make setup
   ```

---

### Issue: "No space left on device"

**Symptoms**:
- Docker build fails
- Error: "no space left on device"

**Causes**:
- Docker disk usage too high
- System disk full

**Solutions**:

1. **Check disk space**:
   ```bash
   df -h  # macOS/Linux
   ```

2. **Clean Docker resources**:
   ```bash
   # Remove unused containers, images, volumes
   docker system prune -a -f --volumes

   # Or selectively:
   docker container prune -f
   docker image prune -a -f
   docker volume prune -f
   ```

3. **Increase Docker disk limit** (Docker Desktop):
   - Docker Desktop → Preferences → Resources → Disk image size
   - Increase to 60GB+

---

## IRIS Connection Issues

### Issue: "Unable to connect to IRIS"

**Symptoms**:
- Python script can't connect to database
- Error: "Connection refused" or "Connection timeout"

**Causes**:
- IRIS not running
- Wrong connection parameters
- Network issues

**Solutions**:

1. **Verify IRIS is running**:
   ```bash
   # Check container status
   docker ps | grep iris

   # Check IRIS logs
   docker logs integratedml-custom-models-iris

   # Test IRIS directly
   docker exec -it integratedml-custom-models-iris iris session iris
   ```

2. **Check connection parameters**:
   ```bash
   # View .env file
   cat .env

   # Verify:
   # IRIS_HOST=localhost
   # IRIS_PORT=1972
   # IRIS_NAMESPACE=USER
   ```

3. **Test connection**:
   ```bash
   # Use test script
   python -c "from shared.database import test_connection; test_connection()"

   # Or manually:
   python <<EOF
   import iris
   conn = iris.connect("localhost", 1972, "USER", "demo", "demo")
   print("✅ Connection successful")
   conn.close()
   EOF
   ```

---

### Issue: "Authentication failed"

**Symptoms**:
- Error: "Invalid username or password"
- Cannot login to Management Portal

**Causes**:
- Wrong credentials in .env
- IRIS security configuration changed

**Solutions**:

1. **Verify credentials**:
   ```bash
   # Default credentials
   # Username: demo
   # Password: demo

   # Check .env file
   cat .env | grep IRIS_USERNAME
   cat .env | grep IRIS_PASSWORD
   ```

2. **Reset IRIS password** (if using Docker):
   ```bash
   # Recreate container with clean state
   make clean
   make setup
   # Default demo/demo credentials will work
   ```

3. **Check IRIS security settings**:
   ```bash
   # Connect to IRIS
   docker exec -it integratedml-custom-models-iris iris session iris

   # Check security settings
   USER> do ##class(%SYSTEM.Security.Users).Get("demo", .properties)
   USER> zwrite properties
   ```

---

## Model Loading Issues

### Issue: "Model not found"

**Symptoms**:
- SQL error: "Model 'MyModel' not found"
- Error during `TRAIN MODEL` or `PREDICT()`

**Causes**:
- Model class file not in correct location
- Model class name mismatch
- IRIS not restarted after model deployment

**Solutions**:

1. **Verify model file exists**:
   ```bash
   # Check model file location
   docker exec -it integratedml-custom-models-iris ls -la /opt/irisapp/data/mgr/python/custom_models/classifiers/

   # Should see your model .py file
   ```

2. **Verify model class name**:
   ```python
   # In your model file, check class name
   class MyCustomModel(ClassificationModel):  # This name must match
       def __init__(self, **kwargs):
           ...
   ```

3. **Copy model if missing**:
   ```bash
   # Copy model to IRIS
   docker cp demos/credit_risk/models/credit_risk_classifier.py \
       integratedml-custom-models-iris:/opt/irisapp/data/mgr/python/custom_models/classifiers/
   ```

4. **Restart IRIS** (required after model changes):
   ```bash
   docker restart integratedml-custom-models-iris

   # Wait for IRIS to be ready (~30 seconds)
   sleep 30
   ```

---

### Issue: "Module 'iris_automl' not found"

**Symptoms**:
- Error: "No module named 'iris_automl'"
- `TRAIN MODEL` fails with import error

**Causes**:
- IntegratedML package not installed
- Symlink missing

**Solutions**:

1. **Check symlink**:
   ```bash
   # Verify symlink exists
   docker exec -it integratedml-custom-models-iris \
       ls -la /opt/irisapp/data/mgr/python/iris_automl

   # Should show symlink to /usr/irissys/mgr/python/iris_automl
   ```

2. **Create symlink**:
   ```bash
   docker exec -it integratedml-custom-models-iris bash
   ln -sf /usr/irissys/mgr/python/iris_automl /opt/irisapp/data/mgr/python/iris_automl
   exit

   docker restart integratedml-custom-models-iris
   ```

3. **Reinstall IntegratedML package**:
   ```bash
   docker exec -it integratedml-custom-models-iris bash
   python -m pip install --index-url https://registry.intersystems.com/pypi/simple \
       --no-cache-dir --target /usr/irissys/mgr/python intersystems-iris-automl
   exit

   docker restart integratedml-custom-models-iris
   ```

---

### Issue: "AttributeError: missing required method"

**Symptoms**:
- Error: "AttributeError: 'MyModel' object has no attribute 'fit'"
- Model training fails

**Causes**:
- Model doesn't implement required methods
- Model doesn't inherit from base class

**Solutions**:

1. **Inherit from base class**:
   ```python
   # ✅ CORRECT
   from shared.models.classification import ClassificationModel

   class MyModel(ClassificationModel):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Your code here

       def fit(self, X, y):
           # Your training code
           return self

       def predict(self, X):
           # Your prediction code
           return predictions
   ```

2. **Implement all required methods**:
   ```python
   # Required methods for classification:
   # - __init__(self, **kwargs)
   # - fit(self, X, y)
   # - predict(self, X)
   # - predict_proba(self, X)  # Optional but recommended
   # - get_params(self, deep=True)
   # - set_params(self, **params)
   ```

3. **Test model outside IRIS**:
   ```bash
   # Test model with pytest
   pytest demos/*/tests/test_*model*.py -v
   ```

---

## Training Issues

### Issue: Training times out

**Symptoms**:
- `TRAIN MODEL` command never completes
- SQL client connection timeout

**Causes**:
- Model training takes too long (>30 minutes)
- Large dataset (>1M rows)
- Complex model (deep learning, etc.)

**Solutions**:

1. **Reduce dataset size for testing**:
   ```sql
   -- Use subset of data
   CREATE MODEL MyModel
   PREDICTING (target)
   FROM (SELECT TOP 10000 * FROM LargeTable)
   ```

2. **Use pre-trained model**:
   ```python
   class MyModel(ClassificationModel):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Load pre-trained model
           if 'pretrained_path' in kwargs:
               self.model = joblib.load(kwargs['pretrained_path'])
               self._is_fitted = True

       def fit(self, X, y):
           if self._is_fitted:
               return self  # Skip training
           # Normal training...
   ```

3. **Increase SQL timeout** (advanced):
   ```bash
   # In IRIS terminal
   USER> set ^MyTimeout = 3600  # 1 hour

   # Or increase in connection string
   ```

---

### Issue: "Out of memory" during training

**Symptoms**:
- Error: "MemoryError" or "Out of memory"
- IRIS container crashes during `TRAIN MODEL`

**Causes**:
- Model requires more memory than available
- Large dataset loaded into memory at once

**Solutions**:

1. **Increase Docker memory**:
   - Docker Desktop → Resources → Memory
   - Set to 12GB or 16GB
   - Apply & Restart

2. **Use smaller dataset for training**:
   ```sql
   -- Sample data
   CREATE MODEL MyModel
   PREDICTING (target)
   FROM (SELECT TOP 50000 * FROM LargeTable)
   ```

3. **Implement batch training** (if model supports):
   ```python
   def fit(self, X, y):
       # Train in batches
       batch_size = 10000
       for i in range(0, len(X), batch_size):
           X_batch = X[i:i+batch_size]
           y_batch = y[i:i+batch_size]
           self.model.partial_fit(X_batch, y_batch)
       return self
   ```

---

### Issue: Training fails with "NaN values"

**Symptoms**:
- Error: "Input contains NaN"
- Training fails immediately

**Causes**:
- Dataset has missing values
- Model doesn't handle NaN values

**Solutions**:

1. **Check for missing values**:
   ```sql
   -- Find columns with NULL values
   SELECT COUNT(*) FROM MyTable WHERE column_name IS NULL
   ```

2. **Impute missing values**:
   ```python
   from sklearn.impute import SimpleImputer

   def _engineer_features(self, X):
       # Handle missing values
       imputer = SimpleImputer(strategy='mean')
       X_imputed = imputer.fit_transform(X)
       return X_imputed
   ```

3. **Filter NULL values in SQL**:
   ```sql
   CREATE MODEL MyModel
   PREDICTING (target)
   FROM (
       SELECT * FROM MyTable
       WHERE feature1 IS NOT NULL
       AND feature2 IS NOT NULL
   )
   ```

---

## Prediction Issues

### Issue: Predictions are always the same value

**Symptoms**:
- All predictions are identical
- Model appears to predict only one class

**Causes**:
- Model not properly fitted
- Model state not saved
- Serialization issue

**Solutions**:

1. **Verify model is fitted**:
   ```python
   def predict(self, X):
       if not hasattr(self, 'model') or self.model is None:
           raise ValueError("Model not fitted. Call fit() first.")
       return self.model.predict(X)
   ```

2. **Check model serialization**:
   ```python
   def _get_model_state(self):
       """Save model state"""
       return {
           'model': joblib.dumps(self.model),
           'fitted': True
       }

   def _set_model_state(self, state):
       """Restore model state"""
       self.model = joblib.loads(state['model'])
   ```

3. **Verify model was trained**:
   ```sql
   -- Check model metadata
   SELECT * FROM INFORMATION_SCHEMA.ML_MODELS
   WHERE model_name = 'MyModel'
   ```

---

### Issue: Predictions are very slow

**Symptoms**:
- `SELECT ... PREDICT()` takes minutes
- Prediction latency >1 second per row

**Causes**:
- Model is complex (ensemble, deep learning)
- Large result set
- No batch processing

**Solutions**:

1. **Limit result set**:
   ```sql
   -- Use WHERE clause
   SELECT TOP 1000 id, PREDICT(MyModel) as prediction
   FROM LargeTable
   WHERE date >= CURRENT_DATE - 7
   ```

2. **Optimize model**:
   ```python
   def predict(self, X):
       # Use vectorized operations
       # Avoid loops
       return self.model.predict(X)  # sklearn models are optimized
   ```

3. **Use batch prediction** (if applicable):
   ```sql
   -- Predict in smaller batches
   SELECT id, PREDICT(MyModel) as prediction
   FROM (SELECT TOP 10000 * FROM LargeTable)
   ```

---

## Performance Issues

### Issue: Model training is slow

**Symptoms**:
- `TRAIN MODEL` takes >5 minutes for small dataset
- Training slower than expected

**Causes**:
- Inefficient model implementation
- Large feature engineering overhead
- Unoptimized code

**Solutions**:

1. **Profile model code**:
   ```python
   import time

   def fit(self, X, y):
       start = time.time()
       X_processed = self._engineer_features(X)
       print(f"Feature engineering: {time.time() - start:.2f}s")

       start = time.time()
       self.model.fit(X_processed, y)
       print(f"Model training: {time.time() - start:.2f}s")

       return self
   ```

2. **Optimize feature engineering**:
   ```python
   import numpy as np

   def _engineer_features(self, X):
       # ✅ Use vectorized operations
       features = np.column_stack([
           X['col1'] * X['col2'],  # Vectorized
           X['col1'].apply(lambda x: x**2)  # Use apply only when necessary
       ])

       # ❌ Avoid loops
       # for i in range(len(X)):
       #     features.append(X['col1'][i] * X['col2'][i])

       return features
   ```

3. **Use efficient model**:
   ```python
   # Use LightGBM instead of XGBoost for speed
   from lightgbm import LGBMClassifier

   self.model = LGBMClassifier(n_estimators=100, n_jobs=-1)
   ```

---

## Platform-Specific Issues

### macOS: Rosetta 2 warning

**Symptoms**:
- Warning: "The requested image's platform (linux/amd64) does not match"
- Slow performance on M1/M2 Macs

**Causes**:
- Docker image built for x86_64, running on ARM64 via Rosetta 2

**Solutions**:

1. **Install Rosetta 2** (if not already):
   ```bash
   softwareupdate --install-rosetta --agree-to-license
   ```

2. **Use ARM64 IRIS image** (if available):
   ```yaml
   # In docker-compose.yml
   image: intersystemsdc/iris-community:latest-arm64
   ```

3. **Accept performance trade-off**:
   - Rosetta 2 performance is good enough for development
   - For production on ARM Macs, request ARM64 IRIS image

---

### Linux: Volume permission errors

**Symptoms**:
- Error: "Permission denied" when IRIS writes to volume
- Container crashes with permission error

**Causes**:
- Docker volume has wrong ownership
- IRIS runs as user 51773, but volume owned by different user

**Solutions**:

```bash
# Fix volume permissions
sudo chown -R 51773:51773 ./data

# Or in docker-compose.yml, add user mapping
services:
  iris:
    user: "51773:51773"
```

---

### Windows: Line ending issues

**Symptoms**:
- Bash scripts fail with "bad interpreter"
- Python scripts have syntax errors

**Causes**:
- Windows CRLF line endings instead of Unix LF

**Solutions**:

```bash
# Configure Git to use LF
git config --global core.autocrlf input

# Convert existing files
find . -type f -name "*.sh" -exec dos2unix {} \;
find . -type f -name "*.py" -exec dos2unix {} \;

# Or in WSL2:
sed -i 's/\r$//' filename.sh
```

---

## Demo-Specific Issues

### Credit Risk Demo: "No module named 'sklearn'"

**Symptoms**:
- Demo fails with sklearn import error

**Solutions**:

```bash
# Install scikit-learn
pip install scikit-learn

# Or install all demo requirements
pip install -r requirements.txt
```

---

### Sales Forecasting Demo: "Prophet not found"

**Symptoms**:
- Demo fails with "No module named 'prophet'"

**Solutions**:

```bash
# Install Prophet
pip install prophet

# Or install demo-specific requirements
pip install -r demos/sales_forecasting/requirements.txt
```

---

### DNA Similarity Demo: "Biopython not installed"

**Symptoms**:
- Demo fails with "No module named 'Bio'"

**Solutions**:

```bash
# Install Biopython
pip install biopython

# Or install demo-specific requirements
pip install -r demos/dna_similarity/requirements.txt
```

---

### Fraud Detection Demo: Data generation takes too long

**Symptoms**:
- Demo hangs during data generation
- Takes >5 minutes

**Solutions**:

```bash
# Reduce data volume in .env
FRAUD_DETECTION_SAMPLES=5000  # Instead of 25000

# Regenerate data
make demo-fraud
```

---

## Diagnostic Information Collection

When contacting support, include this information:

### System Information

```bash
# Collect system info
cat > diagnostic_info.txt <<EOF
## System Information
OS: $(uname -s)
OS Version: $(uname -r)
Architecture: $(uname -m)

## Docker Information
Docker Version: $(docker --version)
Docker Compose Version: $(docker-compose --version)

## Python Information
Python Version: $(python3 --version)
Python Path: $(which python3)

## IRIS Information
IRIS Container Status: $(docker ps | grep iris)
IRIS Logs (last 50 lines):
$(docker logs --tail 50 integratedml-custom-models-iris)

## Environment
$(cat .env)

## Error Details
[Paste error message here]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]
EOF

cat diagnostic_info.txt
```

### Collect Logs

```bash
# IRIS logs
docker logs integratedml-custom-models-iris > iris_logs.txt

# Application logs (if any)
docker-compose logs iml_app > app_logs.txt

# Docker system info
docker info > docker_info.txt
docker ps -a > docker_containers.txt
```

---

## Getting Help

**Still stuck after trying these solutions?**

Email: thomas.dyar@intersystems.com

**Include**:
- [ ] What you were trying to do
- [ ] What went wrong (specific error messages)
- [ ] What you've tried from this guide
- [ ] System information (see [Diagnostic Information Collection](#diagnostic-information-collection))
- [ ] Logs (IRIS logs, Docker logs)
- [ ] Screenshots (if applicable)

**Response time**: 1-2 business days during EAP

---

**— The InterSystems Data Platforms Product Team**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-12
**Coverage**: Installation, Docker, IRIS, Models, Training, Prediction, Performance, Platform-specific
