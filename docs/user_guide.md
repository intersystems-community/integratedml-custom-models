# IntegratedML Flexible Model Integration - Complete User Guide

Welcome to the comprehensive guide for getting started with IntegratedML Flexible Model Integration! This guide will take you from installation through running your first machine learning models integrated directly into database workflows.

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
- [Quick Verification](#-quick-verification)
- [Demo Portfolio Overview](#-demo-portfolio-overview)
- [Demo Walkthroughs](#-demo-walkthroughs)
- [Common Issues & Troubleshooting](#-common-issues--troubleshooting)
- [Next Steps](#-next-steps)

## 🖥️ System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended for all demos)
- **Storage**: 2GB free space
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### Optional but Recommended
- **InterSystems IRIS**: For full IntegratedML integration
- **Jupyter**: For interactive notebooks
- **Git**: For cloning and contributing

### Demo-Specific Requirements
- **Credit Risk**: Scikit-learn, pandas, numpy
- **Fraud Detection**: XGBoost, scikit-learn (GPU optional)
- **Sales Forecasting**: Prophet, LightGBM (additional system dependencies)

## ⚡ Installation & Setup

### Option 1: Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/intersystems/integratedml-demos.git
cd integratedml-demos

# Create and activate virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Conda Installation

```bash
# Clone the repository
git clone https://github.com/intersystems/integratedml-demos.git
cd integratedml-demos

# Create conda environment
conda create -n integratedml-demos python=3.9
conda activate integratedml-demos

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 3: Docker Installation

```bash
# Clone and run with Docker
git clone https://github.com/intersystems/integratedml-demos.git
cd integratedml-demos

# Build and run the development environment
docker-compose up -d

# Access Jupyter Lab at http://localhost:8888
```

## ✅ Quick Verification

Let's verify your installation works correctly by running a simple test:

```bash
# Test basic installation
python -c "
import sys
print('✅ Python version:', sys.version)

try:
    import pandas as pd
    import numpy as np
    import sklearn
    print('✅ Core dependencies loaded successfully')
    
    # Test our demo imports
    from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier
    print('✅ Demo models imported successfully')
    
    print('\n🎉 Installation verified! Ready to run demos.')
except ImportError as e:
    print('❌ Import error:', e)
    print('💡 Try: pip install -r requirements.txt')
"
```

### Quick Demo Test

Run the quick start example to ensure everything works:

```bash
# Run the quick start example
python examples/quick_start_example.py
```

**Expected Output:**
```
IntegratedML Flexible Model Integration Demo - Quick Start Examples
========================================================

DEMO 1: Credit Risk Assessment with Custom Feature Engineering
============================================================
Training data shape: (800, 15)
Test data shape: (200, 15)
...
Accuracy: 0.xxx

🎉 All demos completed successfully!
```

## 📊 Demo Portfolio Overview

Our three progressive demos demonstrate different aspects of IntegratedML integration:

### 🟢 Demo 1: Credit Risk Assessment
**Perfect for**: First-time users, understanding custom feature engineering
- **Complexity**: Beginner-friendly
- **Time Commitment**: 15-30 minutes
- **Key Learning**: Custom preprocessing within database context
- **Business Value**: Secure financial data processing

### 🟡 Demo 2: Fraud Detection 
**Perfect for**: Understanding ensemble techniques and real-time processing
- **Complexity**: Intermediate
- **Time Commitment**: 30-45 minutes  
- **Key Learning**: Ensemble orchestration, real-time constraints
- **Business Value**: 67ms latency, 95.4% accuracy validated

### 🔴 Demo 3: Sales Forecasting
**Perfect for**: Advanced users, third-party library integration
- **Complexity**: Advanced
- **Time Commitment**: 45-60 minutes
- **Key Learning**: Prophet + LightGBM hybrid architecture
- **Business Value**: 20%+ forecasting improvement

## 🚀 Demo Walkthroughs

### Demo 1: Credit Risk Assessment

#### Step 1: Navigate to Demo Directory
```bash
cd demos/credit_risk
```

#### Step 2: Generate Sample Data
```bash
# Create realistic credit risk dataset
python data/generate_sample_data.py
```

#### Step 3: Train and Test the Model
```bash
# Launch interactive notebook
jupyter notebook notebooks/credit_risk_demo.ipynb

# OR run the Python script directly
python -m demos.credit_risk.models.credit_risk_classifier
```

#### Step 4: Explore Custom Features
The demo showcases several custom feature engineering techniques:

- **Debt-to-Income Ratios**: Financial health indicators
- **Credit Utilization Scores**: Spending pattern analysis  
- **Risk Interaction Terms**: Complex relationship modeling
- **Domain-Specific Transformations**: Financial industry best practices

#### Step 5: IntegratedML Integration
```sql
-- Example SQL commands for IntegratedML
CREATE MODEL CreditRiskModel PREDICTING (default_risk)
FROM CreditApplications 
USING CustomCreditRiskClassifier(
    enable_debt_ratio=true,
    enable_interaction_terms=true,
    decision_threshold=0.6
);
```

**📖 [Complete Credit Risk Tutorial →](tutorials/tutorial_01_credit_risk.md)**

---

### Demo 2: Fraud Detection

#### Step 1: Navigate and Setup
```bash
cd demos/fraud_detection

# Install additional dependencies if needed
pip install xgboost
```

#### Step 2: Generate Transaction Data
```bash
# Create synthetic fraud transaction dataset
python data/generate_transaction_data.py
```

#### Step 3: Run Performance Benchmarks
```bash
# Verify latency requirements
python scripts/verify_latency_requirements.py

# Expected output:
# ✅ Average Latency: 67ms (Target: ≤100ms)
# ✅ P95 Latency: 89ms (Target: ≤150ms)  
# ✅ Success Rate: 96.8% (Target: ≥90%)
```

#### Step 4: Explore Ensemble Architecture
The fraud detection system combines:

- **Rule-based Detector**: Fast heuristics (~8ms)
- **Anomaly Detection**: IRIS Vector Search integration (~15ms)
- **Neural Network**: Pattern recognition (~12ms)
- **Behavioral Analysis**: Customer profiling (~9ms)

#### Step 5: Real-time Testing
```bash
# Launch interactive demo
jupyter notebook notebooks/fraud_detection_demo.ipynb

# Test ensemble performance
python -m pytest tests/test_performance.py -v
```

**📖 [Complete Fraud Detection Tutorial →](tutorials/tutorial_02_fraud_detection.md)**

---

### Demo 3: Sales Forecasting

#### Step 1: Install Dependencies
```bash
cd demos/sales_forecasting

# Install Prophet and LightGBM
pip install prophet lightgbm

# Note: Prophet may require additional system dependencies
# See troubleshooting section if you encounter issues
```

#### Step 2: Generate Sales Data
```bash
# Create multi-store retail sales dataset
python data/generate_sales_data.py
```

#### Step 3: Train Hybrid Model
```bash
# Launch forecasting notebook
jupyter notebook notebooks/Sales_Forecasting_Demo.ipynb
```

#### Step 4: Explore Hybrid Architecture
The sales forecasting system combines:

- **Prophet Component**: Trend and seasonality detection
- **LightGBM Component**: Feature-rich ML predictions
- **Ensemble Strategy**: Horizon-weighted combination
- **Confidence Intervals**: Business-ready uncertainty quantification

#### Step 5: Production Integration
```sql
-- Example IntegratedML deployment
CREATE MODEL SalesForecastModel PREDICTING (monthly_sales)
FROM HistoricalSales 
USING HybridForecastingModel(
    trend_model='prophet',
    ml_model='lightgbm',
    forecast_horizon=12,
    include_confidence_intervals=true
);
```

**📖 [Complete Sales Forecasting Tutorial →](tutorials/tutorial_03_sales_forecasting.md)**

## 🔧 Common Issues & Troubleshooting

### Installation Issues

#### Problem: pip install fails with dependency conflicts
```bash
# Solution: Use fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

#### Problem: Prophet installation fails
```bash
# On macOS:
brew install cmake
pip install prophet

# On Ubuntu/Debian:
sudo apt-get install python3-dev python3-pip python3-venv
pip install prophet

# On Windows:
# Install Visual C++ Build Tools first, then:
pip install prophet
```

#### Problem: XGBoost GPU support issues
```bash
# Use CPU version (recommended for most users):
pip install xgboost

# For GPU support (advanced users):
pip install xgboost[gpu]
```

### Runtime Issues

#### Problem: "Module not found" errors
```bash
# Ensure package is installed in development mode:
pip install -e .

# Verify PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Problem: Jupyter notebook kernel issues
```bash
# Install Jupyter kernel for your environment:
python -m ipykernel install --user --name=integratedml-demos
```

#### Problem: Memory errors during training
```bash
# Reduce dataset size for testing:
export DEMO_SAMPLE_SIZE=1000

# Or increase system memory allocation
```

### Performance Issues

#### Problem: Fraud detection latency too high
- Check system resources (CPU/Memory usage)
- Verify no other intensive processes running
- Consider reducing ensemble complexity for testing

#### Problem: Sales forecasting taking too long
- Reduce forecast horizon for testing
- Use smaller dataset for initial exploration
- Check Prophet installation (C++ dependencies)

### IntegratedML Integration Issues

#### Problem: SQL model creation fails
- Verify IntegratedML is properly installed
- Check model class imports and paths
- Ensure database connectivity and permissions

## 🎯 Next Steps

### For ML Practitioners
1. **Explore Custom Features**: Study the feature engineering in Demo 1
2. **Build Your Own Model**: Follow [Tutorial 4: Custom Models](tutorials/tutorial_04_custom_models.md)
3. **Performance Optimization**: Review [Architecture Documentation](architecture.md)

### For IntegratedML Users
1. **Production Deployment**: Review [Deployment Guide](deployment.md)
2. **Integration Patterns**: Study [Architecture Overview](architecture.md)
3. **API Reference**: Explore [API Documentation](api_reference.md)

### For Data Scientists
1. **Performance Analysis**: Deep dive into [Performance Benchmarks](performance_benchmarks.md)
2. **Model Comparison**: Run all three demos and compare approaches
3. **Custom Algorithms**: Adapt the frameworks for your specific use cases

### For Open Source Contributors
1. **Development Setup**: Follow the contributor setup in main README
2. **Contributing Guidelines**: Review [CONTRIBUTING.md](../CONTRIBUTING.md)
3. **Issue Reporting**: Use GitHub Issues for bugs and feature requests

## 🆘 Getting Help

- **📖 Documentation**: Complete reference in `docs/` directory
- **🐛 Issues**: [GitHub Issues](https://github.com/intersystems/integratedml-demos/issues)
- **💬 Community**: [InterSystems Developer Community](https://community.intersystems.com/)
- **✉️ Email**: [support@intersystems.com](mailto:support@intersystems.com)

---

**🎉 You're all set!** Choose your starting demo based on your experience level and dive into the world of IntegratedML Flexible Model Integration. Happy coding! 🚀