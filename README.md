# IntegratedML Flexible Model Integration Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/intersystems-community/integratedml-flexible-model-integration/workflows/CI/badge.svg)](https://github.com/intersystems-community/integratedml-flexible-model-integration/actions/workflows/ci.yml)
[![CodeQL](https://github.com/intersystems-community/integratedml-flexible-model-integration/workflows/CodeQL/badge.svg)](https://github.com/intersystems-community/integratedml-flexible-model-integration/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The complete showcase for IntegratedML's Custom Models feature** - demonstrating how Python ML models integrate seamlessly into InterSystems IRIS SQL workflows. This project provides four real-world examples showing how to deploy custom machine learning models directly into database operations using familiar SQL syntax.

🎯 **Key Innovation**: Execute `CREATE MODEL ... USING "your.custom.model"` and `SELECT PREDICT(YourModel)` to bring any Python ML model into SQL - no data movement required!

## Features

- **In-database processing**: Models execute within the database environment
- **Scikit-learn compatibility**: Uses familiar Python ML patterns and interfaces
- **Multiple model types**: Classification, regression, ensemble, and sequence analysis examples
- **Real-time inference**: Includes examples demonstrating prediction capabilities
- **Production deployment**: Comprehensive setup and deployment documentation

## Quick Start

### Prerequisites
- **Docker & Docker Compose** (for IRIS database)
- **Python 3.8+** (for local development)
- **VS Code** (recommended for notebooks)
- At least 4GB RAM for IRIS container

### 🚀 One-Command Demo

```bash
# Experience all four demos with one command!
python run_all_demos.py --quick

# Or run integration tests only
python run_all_demos.py --test-only
```

### 🛠️ Full Setup

```bash
# Clone the repository
git clone https://github.com/intersystems-community/integratedml-flexible-model-integration.git
cd integratedml-flexible-model-integration

# Complete setup (dependencies + IRIS database)
make setup

# Open notebooks in VS Code
make notebooks
```

That's it! 🎉

### Manual Setup (Alternative)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Start IRIS database only
docker-compose up -d iris

# 4. Open notebooks in VS Code
code notebooks/ demos/
```

### Available Notebooks

Explore these comprehensive demos directly in VS Code:

- [**📓 Quick Start**](notebooks/Iris_IntegratedML_Quickstart.ipynb) - Get familiar with IRIS IntegratedML
- [**💳 Credit Risk**](demos/credit_risk/notebooks/01_Credit_Risk_Complete_Demo.ipynb) - Financial risk assessment
- [**🔒 Fraud Detection**](demos/fraud_detection/notebooks/01_Fraud_Detection_Complete_Demo.ipynb) - Real-time fraud prevention
- [**📈 Sales Forecasting**](demos/sales_forecasting/notebooks/01_Sales_Forecasting_Complete_Demo.ipynb) - Revenue prediction
- [**🧬 DNA Similarity**](demos/dna_similarity/notebooks/01_DNA_Similarity_Complete_Demo.ipynb) - Genomic analysis
- [**📊 Time Series**](demos/time_series_native/notebooks/01_Time_Series_Native_Complete_Demo.ipynb) - Native IRIS time series

### 🛠️ Development Commands

```bash
make help           # Show all available commands
make start          # Start IRIS database
make stop           # Stop IRIS database
make test           # Run all tests
make demos          # Run all demo scripts
make status         # Check system status
```

### 🎉 What's New?
**IntegratedML Custom Models Demo Ready!** Complete showcase with:
- ✅ **All 4 demos working** with comprehensive integration tests
- ✅ **One-command experience** via `run_all_demos.py`
- ✅ **Real-world examples** from finance to genomics
- ✅ **Production-ready patterns** with proper error handling
- ✅ **Interactive notebooks** for hands-on learning

**Simplified Development Workflow**: No more complex multi-container setup! Just IRIS database + local Python development in VS Code.

## Demo Examples

The framework includes four demonstration scenarios with varying complexity levels.

### Demo 1: Credit Risk Assessment
**Basic implementation** • **Custom Feature Engineering** • **Estimated setup: 15 minutes**

Financial risk assessment with in-database feature engineering.

- **Use case**: Credit approval automation with sensitive data protection
- **Technical approach**: Custom preprocessing and domain-specific feature transformations
- **Implementation**: Processing data within database boundaries for compliance
- **[View Tutorial](docs/tutorials/tutorial_01_credit_risk.md)** • **[View Demo](demos/credit_risk/)**

```sql
CREATE MODEL CreditRiskModel PREDICTING (default_risk)
FROM CreditApplications USING CustomCreditRiskClassifier
```

### Demo 2: Real-time Fraud Detection
**Intermediate complexity** • **Ensemble Methods** • **Estimated setup: 15 minutes**

Transaction fraud detection using ensemble model techniques.

- **Use case**: Real-time transaction fraud detection during payment processing
- **Technical approach**: Ensemble model orchestration with weighted voting strategies
- **Implementation**: Includes real-time prediction examples with latency benchmarks
- **[View Tutorial](docs/tutorials/tutorial_02_fraud_detection.md)** • **[View Demo](demos/fraud_detection/)**

```sql
CREATE MODEL FraudDetectionEnsemble PREDICTING (is_fraud)
FROM TransactionStream USING EnsembleFraudDetector(
    voting='weighted', confidence_threshold=0.8
)
```

### Demo 3: Sales Forecasting
**Advanced implementation** • **Third-party Integration** • **Estimated setup: 20 minutes**

Sales forecasting using hybrid model architecture with Prophet and LightGBM integration.

- **Use case**: Sales forecasting for inventory planning and budget allocation
- **Technical approach**: Hybrid model architecture combining Prophet and LightGBM
- **Implementation**: 12-month forecasts with confidence intervals
- **[View Tutorial](docs/tutorials/tutorial_03_sales_forecasting.md)** • **[View Demo](demos/sales_forecasting/)**

```sql
CREATE MODEL SalesForecastModel PREDICTING (monthly_sales)
FROM HistoricalSales USING HybridForecastingModel(
    trend_model='prophet', ml_model='lightgbm',
    forecast_horizon=12, include_confidence_intervals=true
)
```

### Demo 4: DNA Similarity Analysis
**Advanced complexity** • **Sequence Analysis** • **Estimated setup: 25 minutes**

Bioinformatics sequence analysis using Levenshtein distance and k-mer techniques.

- **Use case**: DNA sequence similarity analysis for genomics research and diagnostics
- **Technical approach**: Sequence alignment algorithms and specialized distance metrics
- **Implementation**: DNA matching with optimized sequence processing
- **[View Tutorial](docs/tutorials/tutorial_04_dna_similarity.md)** • **[View Demo](demos/dna_similarity/)**

```sql
CREATE MODEL DNASimilarityModel PREDICTING (similarity_score)
FROM GeneticSequences USING DNASimilarityAnalyzer(
    algorithm='levenshtein', k_mer_size=3,
    similarity_threshold=0.8
)
```

## Project Structure

```
integratedml-flexible-model-integration/
├── README.md                    # Main project overview and quick start
├── LICENSE                      # MIT License
├── requirements.txt             # Core dependencies
├── setup.py                     # Package installation
├── demos/                       # Individual demo implementations
│   ├── credit_risk/             # Demo 1: Credit Risk Assessment
│   ├── fraud_detection/         # Demo 2: Fraud Detection Ensemble
│   ├── sales_forecasting/       # Demo 3: Sales Forecasting
│   └── dna_similarity/          # Demo 4: DNA Similarity Analysis
├── shared/                      # Common utilities and base classes
│   ├── models/                  # Base model interfaces
│   ├── utils/                   # Data processing utilities
│   ├── data/                    # Dataset utilities
│   └── testing/                 # Testing framework
├── docs/                        # Comprehensive documentation
├── examples/                    # Quick start examples
└── tests/                       # Integration and shared tests
```

## Development Setup

### For Contributors

```bash
# Clone and install in development mode
git clone https://github.com/intersystems-community/integratedml-flexible-model-integration.git
cd integratedml-flexible-model-integration
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
flake8 .
```

### Docker Development Environment

```bash
# Build and run the complete development environment
docker-compose up --build -d

# Verify IRIS database is running
docker-compose logs iris

# Access services
open http://localhost:52773/csp/sys/UtilHome.csp  # IRIS Management Portal
open http://localhost:8888                        # Jupyter Lab
open http://localhost:8080                        # Application API

# Run end-to-end demos with live database
python run_credit_risk_demo.py
python run_fraud_detection_demo.py
python run_sales_forecasting_demo.py
python run_dna_similarity_demo.py
```

**Features in this release:**
- IRIS Community Edition database with IntegratedML
- Model training and deployment in database
- End-to-end demo scripts showing complete workflows
- Docker-based development environment for consistent setup
- Data persistence and model lifecycle management

See **[Docker Setup Guide](docs/DOCKER_SETUP.md)** for detailed instructions.

## Documentation

### Getting Started
- **[User Guide](docs/user_guide.md)** - Installation, setup, and walkthrough
- **[Tutorial 1: Credit Risk](docs/tutorials/tutorial_01_credit_risk.md)** - Basic implementation
- **[Tutorial 2: Fraud Detection](docs/tutorials/tutorial_02_fraud_detection.md)** - Ensemble techniques
- **[Tutorial 3: Sales Forecasting](docs/tutorials/tutorial_03_sales_forecasting.md)** - Advanced integration
- **[Tutorial 4: Custom Models](docs/tutorials/tutorial_04_custom_models.md)** - Build your own

### Technical Reference
- **[Architecture Overview](docs/architecture.md)** - System design and integration patterns
- **[API Reference](docs/api_reference.md)** - Complete interface documentation
- **[Performance Benchmarks](docs/performance_benchmarks.md)** - Metrics and analysis
- **[Deployment Guide](docs/deployment.md)** - Production strategies and scaling

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov=demos

# Run performance benchmarks
pytest tests/performance/ -v

# Run specific demo tests
pytest demos/credit_risk/tests/
```

## Performance Information

The following metrics are included for reference based on testing with the provided demo configurations:

| Demo | Key Metrics | Test Results | 
|------|-------------|--------------|
| **Credit Risk** | Accuracy vs Baseline | Within 5% of sklearn baseline |
|  | Prediction Latency | <50ms average |
|  | Setup Complexity | ~15 minutes end-to-end |
| **Fraud Detection** | Ensemble Accuracy | 95.4% detection rate |
|  | Real-time Latency | 67ms average (P95: 89ms) |
|  | Peak Throughput | 6,250 TPS sustained |
|  | Accuracy vs Individual | 10-15% improvement over individual models |
| **Sales Forecasting** | MAPE vs Baseline | 20%+ improvement over naive baselines |
|  | Forecast Latency | <5s for 12-month forecast |
|  | Integration Setup | ~20 minutes with dependencies |

> See **[Performance Benchmarks](docs/performance_benchmarks.md)** for complete methodology and test conditions.

## Contributing

This is a community-driven project. Contributions are welcome from the InterSystems community.

**Read our [Contributing Guide](CONTRIBUTING.md) for details on:**
- Development setup and workflows
- Code standards and review process
- Testing requirements and CI/CD integration
- Community guidelines and communication

### Ways to Contribute

#### Report Issues
- [Bug Reports](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=bug_report.md)
- [Feature Requests](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=feature_request.md)
- [Documentation](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=documentation_improvement.md)
- [Demo Requests](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=demo_request.md)

#### Code Contributions
1. Fork the repository
2. Clone your fork locally
3. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Make your changes following our [Code Standards](CONTRIBUTING.md#code-standards)
5. Test your changes (`pytest` + demo validation)
6. Commit using [Conventional Commits](https://conventionalcommits.org/) (`git commit -m 'feat: add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Submit a [Pull Request](https://github.com/intersystems-community/integratedml-flexible-model-integration/pulls)

#### Community Engagement
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/intersystems-community/integratedml-flexible-model-integration/discussions)
- **Community Forum**: Participate in [InterSystems Developer Community](https://community.intersystems.com/)
- **Code Review**: Help review [open Pull Requests](https://github.com/intersystems-community/integratedml-flexible-model-integration/pulls)

## Support & Community

### Links
- **Documentation**: [integratedml-demos.readthedocs.io](https://integratedml-demos.readthedocs.io/)
- **Discussions**: [GitHub Discussions](https://github.com/intersystems-community/integratedml-flexible-model-integration/discussions)
- **Issues**: [GitHub Issues](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues)
- **Community Forum**: [InterSystems Developer Community](https://community.intersystems.com/)
- **Email**: [support@intersystems.com](mailto:support@intersystems.com)

### Getting Help
- New to IntegratedML? Start with our [User Guide](docs/user_guide.md)
- Questions? Check [GitHub Discussions](https://github.com/intersystems-community/integratedml-flexible-model-integration/discussions) first
- Found a bug? Please [report it](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=bug_report.md)
- Need a feature? Submit a [feature request](https://github.com/intersystems-community/integratedml-flexible-model-integration/issues/new?template=feature_request.md)
- Contributing issues? See our [Contributing Guide](CONTRIBUTING.md)

### Project Status
[![CI Status](https://github.com/intersystems-community/integratedml-flexible-model-integration/workflows/CI/badge.svg)](https://github.com/intersystems-community/integratedml-flexible-model-integration/actions)
[![Security](https://github.com/intersystems-community/integratedml-flexible-model-integration/workflows/CodeQL/badge.svg)](https://github.com/intersystems-community/integratedml-flexible-model-integration/security)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About InterSystems

[InterSystems](https://www.intersystems.com/) provides data platform technology used by institutions in healthcare, government, and business. The technology enables real-time insights from data to solve business problems.
