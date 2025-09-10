#!/bin/bash

# Install IntegratedML Python packages in IRIS Python environment
echo "Installing IntegratedML Python packages..."

# Use IRIS's embedded Python to install packages
# IRIS has its own Python environment at /usr/irissys/mgr/python/
export PYTHONPATH="/usr/irissys/mgr/python:/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# Install packages using IRIS's Python configuration
python3 -m pip install --target /usr/irissys/mgr/python \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    joblib \
    xgboost \
    lightgbm \
    mlxtend \
    imbalanced-learn \
    optuna \
    catboost

# Set proper permissions
chown -R irisowner:irisowner /usr/irissys/mgr/python/

echo "IntegratedML Python packages installed successfully!"

# Verify installation
echo "Verifying package installation..."
python3 -c "import sys; sys.path.insert(0, '/usr/irissys/mgr/python'); import sklearn, pandas, numpy; print('Core ML packages verified!')"

echo "IntegratedML setup completed!"