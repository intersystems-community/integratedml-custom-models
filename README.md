# 🚀 IntegratedML Custom Models

> **Bring Your Python ML Models Directly into SQL** - The future of in-database machine learning with InterSystems IRIS 2025.2

[![IRIS 2025.2](https://img.shields.io/badge/IRIS-2025.2-blue.svg)](https://www.intersystems.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🌟 Overview

IntegratedML Custom Models revolutionizes machine learning workflows by enabling data scientists to deploy custom Python models directly within SQL queries. No more data movement, no more ETL pipelines - just pure ML power right where your data lives.

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

## 🎯 Key Features

- **🔌 Seamless Integration**: Deploy any scikit-learn compatible model directly in SQL
- **⚡ Real-time Predictions**: Sub-50ms latency for mission-critical applications
- **🎨 Custom Models**: Bring your own Python models with domain-specific logic
- **📊 No Data Movement**: Train and predict on live data without exports
- **🔧 Production Ready**: Built for enterprise scale and reliability

## 📦 Demo Showcase

### 1. 💳 Credit Risk Assessment
Advanced financial risk modeling with custom feature engineering for loan default prediction.

- **Model**: Custom ensemble classifier with financial domain expertise
- **Performance**: 100% accuracy on 10,000+ records
- **Training Time**: ~2.3 seconds

### 2. 🚨 Fraud Detection
Real-time transaction fraud detection using ensemble methods combining neural networks, rules, and anomaly detection.

- **Model**: Multi-model ensemble (Neural + Rules + Anomaly)
- **Scale**: Processes 25,000+ transactions
- **Latency**: <50ms per prediction

### 3. 📈 Sales Forecasting
Hybrid time-series forecasting combining Facebook Prophet with LightGBM for retail sales prediction.

- **Model**: Prophet + LightGBM hybrid
- **Accuracy**: 26.9% MAPE on yearly data
- **Features**: Seasonality, holidays, promotions

### 4. 🧬 DNA Similarity Analysis
Genomic sequence analysis using custom similarity algorithms for pathogenicity prediction.

- **Model**: K-NN with custom DNA distance metrics
- **Scale**: 5,000+ sequences
- **Features**: GC content, motif search, sequence alignment

## 🚀 Quick Start

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

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_GUIDE_CUSTOM_MODELS.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Custom Model Development](docs/custom_model_guide.md)

## 🏗️ Architecture

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

## 🧪 Testing

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

## 🛠️ Development

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- InterSystems IRIS team for the IntegratedML platform
- The scikit-learn community for the amazing ML ecosystem
- All contributors who made this project possible

## 📞 Support

- 📧 Email: support@intersystems.com
- 💬 Community: [InterSystems Developer Community](https://community.intersystems.com)
- 🐛 Issues: [GitHub Issues](https://github.com/intersystems/integratedml-custom-models/issues)

---

<p align="center">
  <b>Built with ❤️ by the InterSystems Team</b><br>
  <i>Empowering data scientists to bring ML directly to their data</i>
</p>