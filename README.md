# IntegratedML Custom Models

> Deploy custom Python ML models directly within SQL queries using InterSystems IRIS 2025.2

[![IRIS 2025.2](https://img.shields.io/badge/IRIS-2025.2-blue.svg)](https://www.intersystems.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![EAP Status](https://img.shields.io/badge/status-Early%20Access%20Program-orange.svg)](docs/EAP_GUIDE.md)

---

## 🚀 Early Access Program (EAP)

**Welcome EAP Participants!** The _IntegratedML Custom Models_ will be General Availability (GA)
for the InterSystems IRIS 2026.1 release. Until then, this repository will be the source of documentation
and information about the feature.

### Getting Started with EAP
1. **[Read the EAP Guide](docs/EAP_GUIDE.md)** - Understand the program, timeline, and expectations
2. **[Install Custom Models](docs/INSTALLATION.md)** - Complete installation in <30 minutes (target)
3. **[Check Known Issues](docs/EAP_KNOWN_ISSUES.md)** - Review current limitations before reporting bugs
4. **[Review Roadmap](docs/EAP_ROADMAP.md)** - See what's coming from EAP to GA

### How to Provide Feedback
Your feedback directly shapes the final product! Choose your preferred channel:

- **Survey** (recommended): Survey links provided by Data Platforms Product Team
- **Email**: [thomas.dyar@intersystems.com](mailto:thomas.dyar@intersystems.com)
- **GitHub Issues** (if enabled): Technical bugs and feature requests

**Response time**: 1-2 business days during EAP

For questions or support, see [EAP FAQ](docs/EAP_FAQ.md) or email thomas.dyar@intersystems.com.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Demo Applications](#demo-applications)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
  - [Core Documentation](#-core-documentation-docs)
  - [Demo Applications](#-demo-applications-demos)
  - [Feature Specifications](#-feature-specifications-specs)
- [Architecture](#architecture)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

**IntegratedML Custom Models** extends InterSystems IRIS IntegratedML with a powerful new capability: deploy your own Python models directly within SQL queries. While IntegratedML has provided automated ML for years, this feature gives data scientists full control—custom preprocessing, any scikit-learn compatible model, and third-party libraries like Prophet or LightGBM—all executing in-database without data movement.

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

### 5. AI Functions (SQL-native generative AI)
Brings SingleStore-style `AI_COMPLETE` / `AI_SENTIMENT` / `AI_TRANSLATE` /
`EMBED_TEXT` / `AI_SUMMARIZE` / `AI_CLASSIFY` / `AI_EXTRACT` primitives into
IRIS as IntegratedML Custom Models. Each AI Function is a self-contained
`IRISModel` callable from SQL via `PREDICT(...)`.

- **Models**: 7 IRISModel classes (one per AI Function)
- **Backends**: Offline-by-default (lexicons, TF-IDF, regex), Anthropic Claude for `AI_COMPLETE` when `ANTHROPIC_API_KEY` is set
- **Use cases**: support-ticket triage, real-time fraud detection, customer churn prediction
- See [demos/ai_functions/README.md](demos/ai_functions/README.md)

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

# AI Functions (SQL-native generative AI primitives)
make demo-ai-functions
```

## Documentation

This project's documentation is organized into three main areas:

### 🔶 EAP Documentation (Start Here!)
Essential guides for Early Access Program participants:
- **[EAP Guide](docs/EAP_GUIDE.md)** - Program overview, timeline, feedback channels
- **[Installation Guide](docs/INSTALLATION.md)** - Platform-specific setup (macOS primary, Linux/Windows secondary)
- **[Known Issues](docs/EAP_KNOWN_ISSUES.md)** - Current limitations and workarounds
- **[EAP Roadmap](docs/EAP_ROADMAP.md)** - Features from EAP to GA (2026.1)
- **[EAP FAQ](docs/EAP_FAQ.md)** - Frequently asked questions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### 📚 Core Documentation ([`docs/`](docs/))
Cross-cutting technical documentation for the entire project:
- **[Quick Start Guide](docs/QUICK_GUIDE_CUSTOM_MODELS.md)** - Get started in under 5 minutes
- **[User Guide](docs/user_guide.md)** - Step-by-step usage instructions
- **[Architecture](docs/architecture.md)** - System design, base class hierarchy, integration patterns
- **[API Reference](docs/api_reference.md)** - Complete API documentation for all model classes
- **[Deployment](docs/deployment.md)** - Production deployment strategies and configuration

### 🎯 Demo Applications ([`demos/`](demos/))
Working examples with demo-specific setup instructions:
- **[Credit Risk](demos/credit_risk/)** - Financial risk modeling with custom feature engineering
- **[Fraud Detection](demos/fraud_detection/)** - Transaction fraud detection using ensemble methods
- **[Sales Forecasting](demos/sales_forecasting/)** - Time-series forecasting with Prophet + LightGBM
- **[DNA Similarity](demos/dna_similarity/)** - Sequence analysis with custom distance metrics

### 📋 Feature Specifications ([`specs/`](specs/))
Design documents and implementation plans for new features:
- Feature specifications with user stories and acceptance criteria
- Implementation plans with architecture decisions
- Task breakdowns and validation results

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
