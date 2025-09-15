# IntegratedML Custom Models

> Deploy custom Python ML models directly within SQL queries using InterSystems IRIS 2025.2

[![IRIS 2025.2](https://img.shields.io/badge/IRIS-2025.2-blue.svg)](https://www.intersystems.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

IntegratedML Custom Models allows you to deploy custom Python models directly within SQL queries. This feature enables in-database machine learning without data movement, making it easier to integrate ML models into existing database workflows.

```sql
-- Train your custom Python model with a single SQL command
CREATE MODEL CreditRiskAssessment
PREDICTING (default_risk)
FROM CreditApplications
USING {
    "model_name": "CustomCreditRiskClassifier",
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_risk_scoring": 1
    }
}

-- Get predictions instantly
SELECT customer_id,
       PREDICT(CreditRiskAssessment) as risk_score
FROM NewApplications
```

## Key Features

- **SQL Integration**: Deploy scikit-learn compatible models directly in SQL
- **Low Latency**: Sub-50ms prediction latency for real-time applications
- **Custom Models**: Use your own Python models with domain-specific logic
- **In-Database Processing**: Train and predict without data exports
- **Scalable**: Designed for production workloads

## Demo Applications

### 1. Credit Risk Assessment
Financial risk modeling with custom feature engineering for loan default prediction.

- **Model**: Custom ensemble classifier
- **Test Data**: 10,000 records
- **Training Time**: ~2.3 seconds

### 2. Fraud Detection
Transaction fraud detection using ensemble methods.

- **Model**: Multi-model ensemble (Neural + Rules + Anomaly)
- **Test Data**: 25,000 transactions
- **Latency**: <50ms per prediction

### 3. Sales Forecasting
Time-series forecasting combining Prophet with LightGBM.

- **Model**: Prophet + LightGBM hybrid
- **Accuracy**: 26.9% MAPE
- **Features**: Seasonality, holidays

### 4. DNA Similarity Analysis
Sequence analysis using custom similarity algorithms.

- **Model**: K-NN with custom distance metrics
- **Test Data**: 5,000 sequences
- **Features**: GC content, motif search

## Quick Start

### Prerequisites

- InterSystems IRIS 2025.2+
- Python 3.8+
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/intersystems/integratedml-custom-models.git
cd integratedml-custom-models

# Setup environment (installs dependencies + starts IRIS)
make setup

# Run all demos
make demos
```

### Running Individual Demos

```bash
# Credit Risk Assessment
make demo-credit

# Fraud Detection
make demo-fraud

# Sales Forecasting
make demo-sales

# DNA Similarity
make demo-dna
```

## Documentation

- [Quick Start Guide](docs/QUICK_GUIDE_CUSTOM_MODELS.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SQL Interface                         │
│  CREATE MODEL | TRAIN MODEL | VALIDATE | PREDICT()          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    IntegratedML Engine                       │
│  • Model Management  • Parameter Handling  • Serialization   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Custom Python Models                        │
│  • Scikit-learn Compatible  • Domain-Specific Logic         │
│  • Feature Engineering      • Custom Algorithms             │
└─────────────────────────────────────────────────────────────┘
```

## Testing

```bash
# Run all tests
make test

# Run E2E test with enhanced data volumes
python tests/test_all_demos_e2e.py

# Run specific demo tests
pytest demos/credit_risk/tests/ -v
pytest demos/fraud_detection/tests/ -v
```

### Test Results (Latest)

| Demo | Data Volume | Training Time | Performance |
|------|-------------|---------------|-------------|
| Credit Risk | 10,000 records | 2.3s | 100% accuracy |
| Fraud Detection | 25,000 transactions | 11.7s | 192 flagged |
| Sales Forecasting | 365 days × 5 stores | 0.4s | 26.9% MAPE |
| DNA Similarity | 5,000 sequences | 1.7s | 50.5% accuracy |

## Development

### Project Structure

```
integratedml-custom-models/
├── demos/                    # Demo applications
│   ├── credit_risk/         # Credit risk assessment
│   ├── fraud_detection/     # Fraud detection system
│   ├── sales_forecasting/   # Time series forecasting
│   └── dna_similarity/      # DNA sequence analysis
├── shared/                  # Shared components
│   ├── models/             # Base model classes
│   ├── database/           # IRIS connection utilities
│   └── utils/              # Helper functions
├── docker/                  # Docker configuration
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suites
└── scripts/                # Utility scripts
```

### Creating Custom Models

1. Extend the base model class:

```python
from shared.models.base import IntegratedMLBaseModel

class MyCustomModel(IntegratedMLBaseModel):
    def fit(self, X, y, **params):
        # Your training logic
        pass

    def predict(self, X):
        # Your prediction logic
        pass
```

2. Deploy to IRIS:

```sql
CREATE MODEL MyModel
PREDICTING (target)
FROM MyTable
USING {
    "model_name": "MyCustomModel",
    "path_to_classifiers": "/path/to/models"
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support with IntegratedML Custom Models:

- [InterSystems Developer Community](https://community.intersystems.com)
- [InterSystems Support](https://www.intersystems.com/support/)
