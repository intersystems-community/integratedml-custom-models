# Contributing to IntegratedML Pluggable Models Demo

Thank you for your interest in contributing to the IntegratedML Pluggable Models Demo project! This is a community-driven project hosted in the **intersystems-community** organization, and we welcome contributors of all skill levels.

[![Contributors](https://img.shields.io/github/contributors/intersystems-community/pluggable_iml.svg)](https://github.com/intersystems-community/pluggable_iml/graphs/contributors)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/intersystems-community/pluggable_iml/pulls)
[![Good First Issues](https://img.shields.io/github/issues/intersystems-community/pluggable_iml/good%20first%20issue.svg)](https://github.com/intersystems-community/pluggable_iml/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)

## 🎯 Project Overview

This project demonstrates IntegratedML's pluggable models capability through four progressive demos:
1. **Credit Risk Assessment** (Beginner) - Custom feature engineering
2. **Fraud Detection Ensemble** (Intermediate) - Real-time multi-model orchestration
3. **Sales Forecasting** (Advanced) - Third-party library integration
4. **DNA Similarity Analysis** (Expert) - Bioinformatics sequence analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) InterSystems IRIS with IntegratedML for full functionality

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/pluggable_iml.git
   cd pluggable_iml
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests**
   ```bash
   pytest
   ```

## 🛠️ Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pre-commit** hooks for automated checks

Run code formatting and linting:
```bash
black .
flake8 .
mypy shared/ demos/
```

### Project Structure

```
pluggable_iml/
├── demos/                  # Individual demo implementations
│   ├── credit_risk/        # Demo 1: Credit Risk Assessment
│   ├── fraud_detection/    # Demo 2: Fraud Detection Ensemble
│   ├── sales_forecasting/  # Demo 3: Sales Forecasting
│   └── dna_similarity/     # Demo 4: DNA Similarity Analysis
├── shared/                 # Common utilities and base classes
│   ├── models/            # Base model interfaces
│   ├── utils/             # Data processing utilities
│   ├── data/              # Dataset utilities
│   └── testing/           # Testing framework
├── examples/              # Quick start examples and templates
├── docs/                  # Documentation
└── tests/                 # Test suites
```

### Adding New Features

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Follow the Existing Patterns**
   - Inherit from appropriate base classes in `shared/models/`
   - Follow the established directory structure
   - Include comprehensive docstrings
   - Add type hints for all public methods

3. **Write Tests**
   - Add unit tests for new functionality
   - Include integration tests for demo workflows
   - Ensure tests pass: `pytest tests/`

4. **Update Documentation**
   - Add docstrings to all public methods
   - Update README files if adding new demos
   - Include examples in docstrings

## 📝 Types of Contributions

### 🐛 Bug Reports

We've created detailed issue templates to help you report issues effectively:

**[🐛 Report a Bug](https://github.com/intersystems-community/pluggable_iml/issues/new?template=bug_report.md)**

Our bug report template includes sections for:
- **Environment details** (Python version, OS, package versions)
- **Steps to reproduce** with clear examples
- **Expected vs actual behavior**
- **Demo-specific impact** assessment
- **IntegratedML configuration** details

### ✨ Feature Requests

**[✨ Request a Feature](https://github.com/intersystems-community/pluggable_iml/issues/new?template=feature_request.md)**

Our feature request template covers:
- **Clear description** and motivation
- **Use case** and business value
- **Implementation approach** suggestions
- **IntegratedML integration** considerations
- **Demo complexity** level (Beginner/Intermediate/Advanced/Expert)

### 📘 Documentation Improvements

**[📘 Improve Documentation](https://github.com/intersystems-community/pluggable_iml/issues/new?template=documentation_improvement.md)**

Help us improve our docs by suggesting:
- **Content improvements** and clarifications
- **Missing documentation** for features
- **Tutorial enhancements**
- **Code example** additions

### 🎯 Demo Requests

**[🎯 Request a Demo](https://github.com/intersystems-community/pluggable_iml/issues/new?template=demo_request.md)**

Suggest new demo scenarios:
- **Industry domain** and use case
- **Technical complexity** level
- **ML algorithms** and techniques
- **Business value** demonstration

### 🔧 Code Contributions

#### Demo Enhancements
- Improve existing model implementations
- Add new feature engineering techniques
- Enhance visualization and reporting
- Optimize performance

#### New Demo Models
If you want to add a new demo model:
1. **Create a new directory** under `demos/`
2. **Follow the established structure** (models/, data/, notebooks/, sql/, tests/)
3. **Inherit from appropriate base classes** in `shared/models/`
4. **Include comprehensive documentation** and examples
5. **Add SQL integration scripts** for IntegratedML

#### Infrastructure Improvements
- Enhance base classes in `shared/models/`
- Improve testing framework in `shared/testing/`
- Add utilities in `shared/utils/`
- Improve CI/CD pipeline

#### Documentation
- Improve existing documentation
- Add tutorials and guides
- Create video walkthroughs
- Translate documentation

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual model components
   - Mock external dependencies
   - Fast execution (< 1 second per test)

2. **Integration Tests** (`tests/integration/`)
   - Test end-to-end workflows
   - Include actual model training
   - May use small datasets

3. **Performance Tests** (`tests/performance/`)
   - Benchmark prediction latency
   - Memory usage profiling
   - Scalability testing

### Writing Tests

```python
import pytest
from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier

class TestCustomCreditRiskClassifier:
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = CustomCreditRiskClassifier()
        assert model.enable_debt_ratio is True
        assert model.decision_threshold == 0.5
    
    def test_parameter_validation(self):
        """Test parameter validation works correctly."""
        with pytest.raises(ValueError):
            CustomCreditRiskClassifier(decision_threshold=1.5)
```

### Test Data

- Use synthetic data for tests when possible
- Keep test datasets small (< 1000 samples)
- Store test data in `tests/data/`
- Document data sources and generation methods

## 🚀 Deployment and Release

### Release Process

1. **Version Numbering**: We follow [Semantic Versioning](https://semver.org/)
   - MAJOR: Breaking changes
   - MINOR: New features, backward compatible
   - PATCH: Bug fixes

2. **Release Checklist**:
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped in setup.py
   - [ ] Git tag created
   - [ ] PyPI package published

### GitHub Actions Workflows

Our comprehensive CI/CD pipeline includes multiple specialized workflows:

#### 🔄 **Continuous Integration** ([.github/workflows/ci.yml](.github/workflows/ci.yml))
- **Multi-platform testing** (Linux, Windows, macOS)
- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11)
- **Code quality checks** (Black, Flake8, MyPy)
- **Security scanning** (Bandit, Safety)
- **Demo validation** for all 4 demos
- **Performance benchmarking** with validation gates
- **Docker integration** testing

#### 🔒 **Security Scanning** ([.github/workflows/codeql.yml](.github/workflows/codeql.yml))
- **CodeQL analysis** for vulnerability detection
- **Multi-language security** scanning (Python, JavaScript)
- **ML-specific security** checks
- **Dependency vulnerability** assessment

#### 📖 **Documentation Validation** ([.github/workflows/docs-validation.yml](.github/workflows/docs-validation.yml))
- **Markdown link checking** across all documentation
- **Code example validation** in docs
- **Demo notebook testing** for accuracy
- **Documentation build** verification

#### 🤖 **Dependency Management** ([.github/dependabot.yml](.github/dependabot.yml))
- **Automated dependency updates** via Dependabot
- **ML-specific grouping** for related packages
- **Conservative versioning** for critical libraries
- **Demo-specific** dependency management

**💡 Tip**: All workflows must pass before PRs can be merged. You can view workflow status on your PR page.

## 📋 Pull Request Process

1. **Create a Clear Title**: Use conventional commit format
   - `feat: add new fraud detection rule engine`
   - `fix: resolve memory leak in ensemble model`
   - `docs: update installation instructions`

2. **Write a Detailed Description**:
   - What changes were made
   - Why these changes were necessary
   - How to test the changes
   - Any breaking changes

3. **Ensure Quality**:
   - [ ] All tests pass
   - [ ] Code is formatted with Black
   - [ ] No linting errors
   - [ ] Documentation updated
   - [ ] Type hints added

4. **Request Review**:
   - Tag relevant maintainers
   - Respond to feedback promptly
   - Make requested changes

## 🏷️ Issue Labels

We use the following labels to organize issues:

- **Type**: `bug`, `enhancement`, `documentation`, `question`
- **Priority**: `high`, `medium`, `low`
- **Demo**: `demo:credit-risk`, `demo:fraud-detection`, `demo:sales-forecasting`
- **Component**: `models`, `utils`, `testing`, `docs`
- **Status**: `good-first-issue`, `help-wanted`, `in-progress`

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- **Be respectful** and considerate in all interactions
- **Be collaborative** and help others learn
- **Be patient** with newcomers and different skill levels
- **Give constructive feedback** and be open to receiving it

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, general questions
- **GitHub Discussions**: Design discussions, brainstorming, help
- **InterSystems Community**: IntegratedML-specific questions

## 🏆 Recognition

Contributors are recognized in several ways:
- **Contributors file**: All contributors listed in CONTRIBUTORS.md
- **Release notes**: Significant contributions highlighted
- **GitHub profile**: Contribution graph and repository connection

## 📚 Learning Resources

### IntegratedML Resources
- [IntegratedML Documentation](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GIML)
- [InterSystems Community](https://community.intersystems.com/)

### Machine Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

### Python Development
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)

## ❓ Getting Help

If you need help:

1. **Check existing documentation** in the `docs/` directory
2. **Search existing issues** for similar problems
3. **Ask in GitHub Discussions** for design questions
4. **Create a new issue** with the `question` label
5. **Contact maintainers** for sensitive issues

## 🙏 Thank You

Thank you for contributing to IntegratedML Pluggable Models Demo! Your contributions help make machine learning more accessible and demonstrate the power of database-integrated ML workflows.

---

**Happy coding!** 🚀