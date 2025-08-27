# Sales Forecasting System - Test Suite

This directory contains a comprehensive test suite for the Sales Forecasting System, providing thorough validation of all components, integration testing, and performance benchmarks.

## Overview

The test suite is designed to ensure the reliability, accuracy, and performance of the hybrid forecasting system that combines Facebook Prophet and LightGBM for advanced sales predictions.

## Test Structure

```
tests/
├── __init__.py                          # Test configuration and utilities
├── README.md                           # This file
├── run_tests.py                        # Main test runner with multiple suite options
├── test_hybrid_forecasting_model.py   # Core model testing with time series validation
├── test_components.py                  # Individual component testing
└── test_integration.py                 # End-to-end pipeline and performance testing
```

## Test Suites

### 1. Unit Tests (`test_components.py`)
Tests individual components in isolation:
- **SalesDataGenerator**: Data generation with seasonal patterns
- **FeatureEngineer**: Time series feature engineering pipeline
- **ForecastEvaluator**: Forecast accuracy and business metrics
- **BusinessIntelligence**: Executive dashboards and KPI analytics

### 2. Model Tests (`test_hybrid_forecasting_model.py`)
Comprehensive testing of the hybrid forecasting model:
- **HybridForecastingModel**: Core functionality, fitting, prediction
- **TimeSeriesValidation**: Walk-forward validation, temporal consistency
- **ModelComponents**: Prophet, LightGBM, seasonal analyzer, trend detector
- **BusinessLogic**: Business intelligence and evaluation components
- **DataQuality**: Data validation and preprocessing

### 3. Integration Tests (`test_integration.py`)
End-to-end pipeline testing:
- **EndToEndPipeline**: Complete forecasting workflow
- **PerformanceBenchmarks**: Training/prediction speed, memory usage
- **CrossValidation**: Time series cross-validation
- **ModelPersistence**: Serialization and loading
- **BusinessIntelligence**: Full BI integration

## Quick Start

### Run All Tests
```bash
# Navigate to tests directory
cd demos/sales_forecasting/tests

# Run complete test suite
python run_tests.py --suite all

# Run with detailed output and report
python run_tests.py --suite all --verbosity 2 --report test_results.txt
```

### Run Specific Test Suites
```bash
# Quick smoke tests (1-2 minutes)
python run_tests.py --suite smoke

# Essential development tests (3-5 minutes)
python run_tests.py --suite quick

# Unit tests only (2-3 minutes)
python run_tests.py --suite unit

# Model tests with time series validation (5-10 minutes)
python run_tests.py --suite model

# Integration and performance tests (10-15 minutes)
python run_tests.py --suite integration

# Performance benchmarks only (5-10 minutes)
python run_tests.py --suite performance
```

### Run Individual Test Files
```bash
# Run specific test file
python test_hybrid_forecasting_model.py
python test_components.py
python test_integration.py

# Run with unittest
python -m unittest test_hybrid_forecasting_model.TestHybridForecastingModel
python -m unittest test_components.TestSalesDataGenerator
```

## Test Configuration

### Environment Setup
```bash
# Install test dependencies
pip install pytest numpy pandas scikit-learn lightgbm

# Optional: Install Prophet for full functionality
pip install prophet

# Set environment variables (optional)
export SALES_TEST_DATA_SIZE=small  # small, medium, large
export SALES_TEST_VERBOSE=true
export SALES_TEST_ARTIFACTS=false
```

### Test Data Sizes
- **Small**: 90 days, 1 store, 1 category (~300 training samples)
- **Medium**: 365 days, 2 stores, 2 categories (~1000 training samples)  
- **Large**: 730 days, 5 stores, 3 categories (~2000 training samples)

## Test Coverage

### Core Functionality Tests
- ✅ Model initialization and configuration
- ✅ Data preprocessing and feature engineering
- ✅ Model training with hybrid architecture
- ✅ Prediction generation and validation
- ✅ Model serialization and persistence

### Time Series Specific Tests
- ✅ Walk-forward cross-validation
- ✅ Seasonal pattern preservation
- ✅ Temporal consistency validation
- ✅ Forecast horizon performance
- ✅ Trend detection and analysis

### Business Logic Tests
- ✅ Forecast accuracy metrics (MAPE, MAE, RMSE, R²)
- ✅ Business impact analysis
- ✅ ROI calculations and KPI tracking
- ✅ Executive dashboard generation
- ✅ Scenario analysis and recommendations

### Integration Tests
- ✅ End-to-end pipeline execution
- ✅ Multi-store/category processing
- ✅ Cross-validation workflows
- ✅ Feature engineering robustness
- ✅ Business intelligence integration

### Performance Tests
- ✅ Training time benchmarks (< 300 seconds)
- ✅ Prediction speed (> 100 predictions/second)
- ✅ Memory usage monitoring (< 1000 MB)
- ✅ Scalability with data size
- ✅ Model reproducibility

## Expected Test Results

### Performance Thresholds
- **Training Time**: < 5 minutes for medium dataset
- **Prediction Speed**: > 100 predictions/second
- **Memory Usage**: < 1 GB during training
- **Forecast Accuracy**: MAPE < 50% (target < 25%)
- **Model Correlation**: R² > 0.3 (target > 0.7)

### Success Criteria
- **Unit Tests**: 100% pass rate
- **Model Tests**: > 95% pass rate (some stochastic variation acceptable)
- **Integration Tests**: > 90% pass rate
- **Performance Tests**: Meet all benchmark thresholds

## Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# Install missing packages
pip install -r ../../../requirements.txt
pip install prophet  # If Prophet tests are failing
```

**Memory Issues**
```bash
# Use smaller test data size
python run_tests.py --suite quick
# Or set environment variable
export SALES_TEST_DATA_SIZE=small
```

**Slow Test Execution**
```bash
# Run smoke tests for quick validation
python run_tests.py --suite smoke

# Skip performance tests
python run_tests.py --suite unit
python run_tests.py --suite model
```

**Prophet Installation Issues**
```bash
# On macOS
brew install cmake
pip install prophet

# On Ubuntu
sudo apt-get install python3-dev
pip install prophet

# Alternative: Use conda
conda install -c conda-forge prophet
```

### Test Debugging

**Verbose Output**
```bash
python run_tests.py --suite all --verbosity 2
```

**Individual Test Debugging**
```bash
# Run specific test with detailed output
python -m unittest test_hybrid_forecasting_model.TestHybridForecastingModel.test_model_fitting -v

# Use Python debugger
python -m pdb test_hybrid_forecasting_model.py
```

**Test Artifacts**
```bash
# Save test artifacts for analysis
export SALES_TEST_ARTIFACTS=true
python run_tests.py --suite integration
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Sales Forecasting Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install prophet
    - name: Run tests
      run: |
        cd demos/sales_forecasting/tests
        python run_tests.py --suite all --report ci_results.txt
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Add to .pre-commit-config.yaml
repos:
- repo: local
  hooks:
  - id: sales-forecast-tests
    name: Sales Forecasting Tests
    entry: python demos/sales_forecasting/tests/run_tests.py --suite smoke
    language: system
    pass_filenames: false
```

## Contributing

### Adding New Tests
1. **Unit Tests**: Add to `test_components.py` for new components
2. **Model Tests**: Add to `test_hybrid_forecasting_model.py` for model functionality
3. **Integration Tests**: Add to `test_integration.py` for end-to-end workflows

### Test Naming Conventions
- Test classes: `Test<ComponentName>`
- Test methods: `test_<functionality_description>`
- Test files: `test_<module_name>.py`

### Test Documentation
- Include docstrings for all test methods
- Document test assumptions and expected behaviors
- Add comments for complex test logic

## Performance Optimization

### Test Execution Speed
- Use smaller datasets for development testing
- Mock external dependencies when possible
- Run critical tests in parallel where safe
- Cache expensive setup operations

### Resource Management
- Clean up temporary files and models
- Monitor memory usage during long test runs
- Use context managers for resource cleanup
- Implement test timeouts for long-running operations

## Reporting

The test suite generates comprehensive reports including:
- **Test Summary**: Pass/fail counts and success rates
- **Performance Metrics**: Timing and resource usage
- **Coverage Analysis**: Feature and code coverage
- **Business Metrics**: Forecast accuracy and business impact
- **Recommendations**: Actions based on test results

Example report output:
```
SALES FORECASTING SYSTEM - TEST REPORT
=====================================
Generated: 2024-01-15 14:30:22

SUMMARY
Total Tests: 127
Passed: 124
Failed: 2
Errors: 1
Success Rate: 97.6%

DETAILED RESULTS
Unit Tests: 45 tests, 0 failures, 0 errors, 0 skipped, 100.0% success
Model Tests: 52 tests, 1 failures, 1 errors, 0 skipped, 96.2% success
Integration Tests: 30 tests, 1 failures, 0 errors, 0 skipped, 96.7% success

🎉 EXCELLENT TEST COVERAGE - SYSTEM READY FOR DEPLOYMENT
```

For questions or issues with the test suite, please refer to the main project documentation or create an issue in the project repository.