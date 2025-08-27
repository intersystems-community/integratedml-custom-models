# Demo 3: Sales Forecasting with Third-party Library Integration

🔴 **Complexity**: Advanced  
🎯 **Focus**: Integrating specialized libraries (Prophet/LightGBM)

## Business Problem

Retail organizations need accurate sales forecasting for inventory planning, budget allocation, and resource management. Traditional forecasting approaches often lack sophisticated seasonality detection or require complex deployment of specialized libraries like Facebook Prophet.

## Solution Overview

This demo showcases **integration of best-of-breed forecasting libraries** within IntegratedML, demonstrating how to bring specialized tools like Prophet and LightGBM into database workflows while managing complex dependencies and hybrid model architectures.

### Key Benefits
- **Accuracy**: 20%+ MAPE improvement over naive forecasting baselines
- **Sophistication**: Advanced seasonality detection and trend analysis
- **Integration**: Seamless deployment of external libraries in database context
- **Business Value**: 12-month rolling forecasts with confidence intervals

## Technical Approach

### Dataset: Multi-store Retail Sales
- **Source**: Synthetic retail sales data with realistic patterns
- **Records**: 3+ years of daily sales data across multiple stores
- **Features**: Historical sales, seasonality, promotions, external factors (holidays, weather)
- **Business Context**: Monthly forecasting for inventory planning and budget allocation

### Hybrid Model Architecture
- **Primary Model**: Facebook Prophet for trend and seasonality detection
- **Secondary Model**: LightGBM for feature-rich predictions with external factors
- **Integration**: Advanced wrapper handling dependencies, serialization, and hybrid predictions
- **Ensemble Strategy**: Weighted combination based on forecast horizon and confidence

### Advanced Capabilities
- **Dependency Management**: Automated handling of Prophet and LightGBM requirements
- **Model Serialization**: Complex state persistence for production deployment
- **Confidence Intervals**: Business-ready uncertainty quantification
- **External Factors**: Holiday effects, promotional impacts, weather integration

## Quick Start

### Prerequisites
```bash
# Ensure you're in the project root
cd ../../

# Install dependencies (including Prophet/LightGBM)
pip install -r requirements.txt

# Note: Prophet may require additional system dependencies
# On macOS: brew install cmake
# On Ubuntu: apt-get install build-essential
```

### Run the Demo

1. **Generate Retail Sales Data**:
   ```bash
   cd demos/sales_forecasting
   python scripts/generate_sales_data.py
   ```

2. **Train Hybrid Forecasting Model**:
   ```bash
   python scripts/train_forecaster.py
   ```

3. **Business Forecasting Dashboard**:
   ```bash
   jupyter notebook notebooks/04_business_dashboard.ipynb
   ```

## Expected User Experience

### SQL Integration
```sql
-- Create the hybrid forecasting model
CREATE MODEL SalesForecastModel PREDICTING (monthly_sales)
FROM HistoricalSales 
USING HybridForecastingModel(
    trend_model='prophet',
    ml_model='lightgbm', 
    forecast_horizon=12,
    include_confidence_intervals=true,
    seasonal_periods=['yearly', 'monthly', 'weekly']
)

-- Generate business forecasts with uncertainty bounds
SELECT store_id, forecast_month,
       PREDICT(SalesForecastModel) as predicted_sales,
       PREDICT(SalesForecastModel WITH 'confidence_lower') as lower_bound,
       PREDICT(SalesForecastModel WITH 'confidence_upper') as upper_bound,
       PREDICT(SalesForecastModel WITH 'trend') as trend_component,
       PREDICT(SalesForecastModel WITH 'seasonal') as seasonal_component
FROM ForecastingInput
WHERE forecast_month BETWEEN '2024-01-01' AND '2024-12-31'
```

### Python Integration
```python
from models.hybrid_forecasting_model import HybridForecastingModel

# Initialize with sophisticated configuration
forecaster = HybridForecastingModel(
    trend_model='prophet',
    ml_model='lightgbm',
    forecast_horizon=12,
    seasonal_periods=['yearly', 'monthly'],
    external_regressors=['holidays', 'promotions', 'weather'],
    confidence_intervals=True
)

# Train with complex time series data
forecaster.fit(
    historical_data=sales_data,
    external_data=external_factors,
    validation_period='2023-01-01'
)

# Generate business-ready forecasts
forecast = forecaster.predict(
    periods=12,
    include_confidence=True,
    include_components=True
)
```

## Files and Structure

```
sales_forecasting/
├── README.md                           # This file
├── models/
│   ├── __init__.py
│   ├── hybrid_forecasting_model.py     # Main hybrid model implementation
│   ├── prophet_component.py            # Prophet integration wrapper
│   ├── lightgbm_component.py           # LightGBM forecasting wrapper
│   ├── ensemble_forecaster.py          # Model combination logic
│   └── dependency_manager.py           # External library management
├── data/
│   ├── README.md                       # Data generation and sources
│   ├── retail_sales_data.csv           # Generated sales time series
│   ├── external_factors.csv            # Holiday, weather, promotion data
│   ├── sales_data_generator.py         # Realistic pattern creation
│   └── external_data_collector.py      # External factor integration
├── notebooks/
│   ├── 01_data_generation.ipynb        # Synthetic sales data creation
│   ├── 02_prophet_exploration.ipynb    # Prophet model analysis
│   ├── 03_lightgbm_features.ipynb      # Feature engineering for ML
│   ├── 04_hybrid_model.ipynb           # Hybrid model development
│   ├── 05_business_dashboard.ipynb     # Executive forecasting dashboard
│   └── 06_integratedml_integration.ipynb # SQL integration demo
├── scripts/
│   ├── generate_sales_data.py          # Automated data generation
│   ├── train_forecaster.py             # Hybrid model training
│   ├── validate_forecasts.py           # Accuracy evaluation
│   ├── benchmark_models.py             # Performance comparison
│   └── deploy_forecaster.py            # IntegratedML deployment
├── sql/
│   ├── create_tables.sql               # Sales data schema setup
│   ├── create_forecasting_model.sql    # IntegratedML model creation
│   ├── business_forecasts.sql          # Business-ready forecast queries
│   └── forecast_accuracy_tracking.sql  # Model performance monitoring
└── tests/
    ├── test_hybrid_model.py            # Hybrid model functionality
    ├── test_prophet_integration.py     # Prophet wrapper tests
    ├── test_lightgbm_integration.py    # LightGBM wrapper tests
    ├── test_dependency_management.py   # External library tests
    └── test_integration.py             # IntegratedML integration tests
```

## Performance Expectations

### Accuracy Targets
- **MAPE Improvement**: 20%+ improvement over naive forecasting baseline
- **Seasonal Detection**: Accurate identification of yearly, monthly, weekly patterns
- **Trend Analysis**: Robust long-term trend forecasting with confidence intervals

### Performance Targets
- **Forecast Generation**: < 5 seconds for 12-month horizon
- **Memory Usage**: Appropriate for typical business datasets (millions of records)
- **Dependency Management**: Reliable installation across environments

### Business Metrics
- **Setup Time**: Complete demo in < 20 minutes (including dependency installation)
- **Business Relevance**: Forecasts suitable for inventory and budget planning
- **Visualization**: Clear, executive-ready forecast presentations

> 📊 **Detailed Performance Metrics**: See our comprehensive [Performance Benchmarks](../../docs/performance_benchmarks.md) for complete accuracy, latency, and throughput measurements across all demos.

## Learning Objectives

By completing this demo, you will understand:

1. **Library Integration**: How to incorporate specialized libraries into IntegratedML
2. **Dependency Management**: Handling complex external dependencies in production
3. **Hybrid Models**: Combining different modeling approaches for superior performance
4. **Time Series Forecasting**: Advanced techniques for business forecasting
5. **Production Deployment**: Deploying complex models in enterprise environments

## Advanced Configuration

### Prophet Configuration
```python
# Sophisticated Prophet setup
prophet_config = {
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'holidays': holiday_dataframe,
    'seasonality_prior_scale': 10.0,
    'changepoint_prior_scale': 0.05
}

forecaster = HybridForecastingModel(
    trend_model='prophet',
    prophet_params=prophet_config
)
```

### LightGBM Integration
```python
# Advanced LightGBM for time series
lightgbm_config = {
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

forecaster = HybridForecastingModel(
    ml_model='lightgbm',
    lightgbm_params=lightgbm_config,
    feature_engineering='advanced'
)
```

### Ensemble Strategy
```python
# Weighted ensemble based on forecast horizon
ensemble_config = {
    'short_term_weight': {'prophet': 0.3, 'lightgbm': 0.7},  # 1-3 months
    'medium_term_weight': {'prophet': 0.5, 'lightgbm': 0.5}, # 4-9 months
    'long_term_weight': {'prophet': 0.7, 'lightgbm': 0.3}    # 10-12 months
}

forecaster = HybridForecastingModel(
    ensemble_strategy='horizon_weighted',
    ensemble_config=ensemble_config
)
```

## Business Integration Patterns

### Inventory Planning
```sql
-- Generate inventory requirements based on sales forecasts
SELECT 
    store_id,
    product_category,
    forecast_month,
    predicted_sales,
    predicted_sales * 1.2 as safety_stock_target,
    CASE 
        WHEN upper_bound - predicted_sales > predicted_sales * 0.3 
        THEN 'HIGH_UNCERTAINTY'
        ELSE 'NORMAL'
    END as forecast_confidence
FROM SalesForecasts
WHERE forecast_month BETWEEN '2024-01-01' AND '2024-12-31'
```

### Budget Planning
```sql
-- Revenue forecasts for financial planning
SELECT 
    fiscal_quarter,
    SUM(predicted_sales * average_margin) as predicted_revenue,
    SUM(lower_bound * average_margin) as conservative_revenue,
    SUM(upper_bound * average_margin) as optimistic_revenue
FROM SalesForecasts sf
JOIN ProductMargins pm ON sf.product_category = pm.category
GROUP BY fiscal_quarter
ORDER BY fiscal_quarter
```

## Dependency Troubleshooting

### Prophet Installation Issues
```bash
# Common Prophet installation problems

# macOS - Missing cmake
brew install cmake

# Ubuntu/Debian - Missing build tools
sudo apt-get install build-essential

# Python - Specific Prophet version
pip install prophet==1.1.4

# Conda alternative
conda install -c conda-forge prophet
```

### LightGBM Performance
```python
# Optimize LightGBM for time series
lightgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_threads': -1,  # Use all available cores
    'force_row_wise': True,  # Better for time series
    'verbose': -1
}
```

## Next Steps

After completing this demo:
- 🟢 **[Demo 1: Credit Risk](../credit_risk/README.md)** - Learn basic custom feature engineering
- 🟡 **[Demo 2: Fraud Detection](../fraud_detection/README.md)** - Understand ensemble model orchestration
- 📖 **[Time Series Guide](../../docs/time-series-forecasting.md)** - Deep dive into forecasting techniques
- 📖 **[Dependency Management](../../docs/dependency-management.md)** - Advanced library integration patterns

## Troubleshooting

### Common Issues

**Issue**: Prophet installation fails
```
Solution: Install system dependencies (cmake, build tools)
Check: System requirements for Prophet compilation
Alternative: Use conda instead of pip for Prophet
```

**Issue**: Forecast accuracy lower than expected
```
Solution: Tune Prophet seasonality parameters and LightGBM features
Check: Data quality and sufficient historical data (min 2 years)
```

**Issue**: Model training takes too long
```
Solution: Optimize LightGBM parameters and reduce Prophet changepoint density
Check: Data size and feature engineering complexity
```

**Issue**: Dependency conflicts between Prophet and other libraries
```
Solution: Use virtual environment with specific package versions
Check: requirements.txt for compatible version specifications
```

**Issue**: Serialization fails for hybrid model
```
Solution: Implement custom serialization for Prophet components
Check: dependency_manager.py for serialization handling
```

## Support

- **Demo-specific issues**: [GitHub Issues](https://github.com/intersystems/integratedml-demos/issues) with `demo:sales-forecasting` label
- **Prophet questions**: [Prophet Documentation](https://facebook.github.io/prophet/)
- **LightGBM questions**: [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- **Time series forecasting**: [Forecasting Best Practices](../../docs/forecasting-best-practices.md)
- **Dependency management**: [Library Integration Guide](../../docs/library-integration.md)