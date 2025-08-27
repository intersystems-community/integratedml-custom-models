# Demo 2: Real-time Fraud Detection with Ensemble Models

🟡 **Complexity**: Intermediate  
🎯 **Focus**: Combining multiple models with real-time scoring capabilities

## Business Problem

Financial institutions need to detect fraudulent transactions in real-time during payment processing. Traditional approaches often use single models with limited accuracy or require external API calls that introduce latency and reliability issues.

## Solution Overview

This demo showcases **ensemble model orchestration within IntegratedML**, demonstrating how to combine diverse model types for superior fraud detection while maintaining sub-100ms latency requirements.

### Key Benefits
- **Accuracy**: 10-15% improvement over best individual model
- **Performance**: Sub-100ms real-time scoring at database scale
- **Reliability**: No external API dependencies for mission-critical decisions
- **Flexibility**: Configurable voting strategies and confidence thresholds

## Technical Approach

### Dataset: Synthetic Credit Card Transactions
- **Source**: Synthetic dataset with realistic fraud patterns
- **Records**: 100,000+ transactions with fraud labels
- **Features**: Transaction amount, merchant category, time patterns, customer behavior
- **Real-time Context**: Streaming transaction data with sub-100ms scoring requirements

### Ensemble Architecture
- **Isolation Forest**: Anomaly detection for unusual transaction patterns
- **XGBoost**: Pattern recognition for complex fraud indicators
- **Rule Engine**: Business logic for known fraud patterns (thresholds, blocklists)
- **Voting Strategy**: Weighted voting with confidence-based thresholds

### Model Implementation
- **Integration**: Custom ensemble class implementing BaseEstimator + ClassifierMixin
- **Orchestration**: Parallel model execution with result aggregation
- **Persistence**: Efficient model serialization for production deployment

## Quick Start

### Prerequisites
```bash
# Ensure you're in the project root
cd ../../

# Install dependencies if not already done
pip install -r requirements.txt
```

### Run the Demo

1. **Generate Synthetic Data**:
   ```bash
   cd demos/fraud_detection
   python scripts/generate_data.py
   ```

2. **Train Ensemble Models**:
   ```bash
   python scripts/train_ensemble.py
   ```

3. **Real-time Performance Testing**:
   ```bash
   jupyter notebook notebooks/03_realtime_testing.ipynb
   ```

## Expected User Experience

### SQL Integration
```sql
-- Create the ensemble model
CREATE MODEL FraudDetectionEnsemble PREDICTING (is_fraud)
FROM TransactionStream 
USING EnsembleFraudDetector(
    voting='weighted', 
    confidence_threshold=0.8,
    enable_rule_engine=true
)

-- Real-time scoring in transaction pipeline
SELECT transaction_id, amount, merchant_category,
       PREDICT(FraudDetectionEnsemble) as fraud_probability,
       PREDICT(FraudDetectionEnsemble WITH 'class') as fraud_decision,
       PREDICT(FraudDetectionEnsemble WITH 'confidence') as model_confidence
FROM LiveTransactions
WHERE transaction_timestamp > NOW() - INTERVAL 1 MINUTE
```

### Python Integration
```python
from models.ensemble_fraud_detector import EnsembleFraudDetector

# Initialize ensemble with custom configuration
ensemble = EnsembleFraudDetector(
    voting='weighted',
    confidence_threshold=0.8,
    models={
        'isolation_forest': {'contamination': 0.1},
        'xgboost': {'n_estimators': 100, 'max_depth': 6},
        'rule_engine': {'enable_amount_rules': True}
    }
)

# Train ensemble (fits all component models)
ensemble.fit(X_train, y_train)

# Real-time prediction with confidence
prediction, confidence = ensemble.predict_with_confidence(transaction_data)
```

## Files and Structure

```
fraud_detection/
├── README.md                           # This file
├── models/
│   ├── __init__.py
│   ├── ensemble_fraud_detector.py      # Main ensemble implementation
│   ├── isolation_forest_model.py       # Anomaly detection component
│   ├── xgboost_model.py                # ML pattern recognition
│   ├── rule_engine.py                  # Business logic rules
│   └── voting_strategies.py            # Ensemble combination logic
├── data/
│   ├── README.md                       # Data generation and sources
│   ├── synthetic_transactions.csv      # Generated transaction data
│   ├── data_generator.py               # Realistic fraud pattern creation
│   └── streaming_simulator.py          # Real-time data simulation
├── notebooks/
│   ├── 01_data_generation.ipynb        # Synthetic data creation
│   ├── 02_individual_models.ipynb      # Component model analysis
│   ├── 03_ensemble_training.ipynb      # Ensemble development
│   ├── 04_realtime_testing.ipynb       # Performance benchmarking
│   └── 05_integratedml_integration.ipynb # SQL integration demo
├── scripts/
│   ├── generate_data.py                # Automated data generation
│   ├── train_ensemble.py               # Ensemble training pipeline
│   ├── benchmark_performance.py        # Latency and accuracy testing
│   └── deploy_ensemble.py              # IntegratedML deployment
├── sql/
│   ├── create_tables.sql               # Transaction schema setup
│   ├── create_ensemble_model.sql       # IntegratedML model creation
│   ├── realtime_scoring.sql            # Real-time prediction queries
│   └── performance_monitoring.sql      # Model performance tracking
└── tests/
    ├── test_ensemble.py                # Ensemble functionality tests
    ├── test_individual_models.py       # Component model tests
    ├── test_performance.py             # Latency and throughput tests
    └── test_integration.py             # IntegratedML integration tests
```

## Performance Expectations

### Accuracy Targets
- **Ensemble Improvement**: 10-15% accuracy improvement over best individual model
- **Precision/Recall**: Balanced performance for both fraud detection and false positive minimization
- **Model Robustness**: Consistent performance across different fraud patterns

### Latency Targets
- **95th Percentile**: < 100ms for real-time scoring
- **Concurrent Requests**: Handle 100+ concurrent prediction requests
- **Memory Footprint**: Linear scaling with ensemble size

### Business Metrics
- **Setup Time**: Complete demo in < 15 minutes
- **Real-time Demonstration**: Clear visualization of sub-100ms performance
- **Configuration Options**: Well-documented with sensible defaults

> 📊 **Detailed Performance Metrics**: See our comprehensive [Performance Benchmarks](../../docs/performance_benchmarks.md) for complete accuracy, latency, and throughput measurements across all demos.

## Learning Objectives

By completing this demo, you will understand:

1. **Ensemble Orchestration**: How to coordinate multiple models within IntegratedML
2. **Real-time Performance**: Achieving sub-100ms latency at database scale
3. **Voting Strategies**: Different approaches to combining model outputs
4. **Confidence Scoring**: Using model uncertainty for decision making
5. **Production Deployment**: Scalable ensemble model deployment patterns

## Advanced Configuration

### Voting Strategies
```python
# Weighted voting based on individual model performance
ensemble = EnsembleFraudDetector(voting='weighted', weights=[0.4, 0.4, 0.2])

# Soft voting using probability distributions
ensemble = EnsembleFraudDetector(voting='soft', threshold=0.7)

# Confidence-based voting with fallback strategy
ensemble = EnsembleFraudDetector(
    voting='confidence_weighted',
    min_confidence=0.6,
    fallback_strategy='conservative'
)
```

### Performance Tuning
```python
# Optimize for latency
ensemble.configure(
    parallel_execution=True,
    cache_predictions=True,
    batch_size=1000
)

# Optimize for accuracy
ensemble.configure(
    enable_feature_selection=True,
    cross_validate_weights=True,
    retrain_interval='daily'
)
```

## Real-time Integration Patterns

### Streaming Data Processing
```sql
-- Create real-time view for fraud scoring
CREATE VIEW FraudScoredTransactions AS
SELECT *,
       PREDICT(FraudDetectionEnsemble) as fraud_score,
       CASE 
         WHEN PREDICT(FraudDetectionEnsemble) > 0.8 THEN 'BLOCK'
         WHEN PREDICT(FraudDetectionEnsemble) > 0.5 THEN 'REVIEW'
         ELSE 'APPROVE'
       END as decision
FROM TransactionStream
```

### Performance Monitoring
```sql
-- Monitor ensemble performance in real-time
SELECT 
    AVG(scoring_latency_ms) as avg_latency,
    MAX(scoring_latency_ms) as max_latency,
    COUNT(*) as predictions_per_second
FROM FraudScoringMetrics
WHERE timestamp > NOW() - INTERVAL 1 HOUR
```

## Next Steps

After completing this demo:
- 🟢 **[Demo 1: Credit Risk](../credit_risk/README.md)** - Learn basic custom feature engineering
- 🔴 **[Demo 3: Sales Forecasting](../sales_forecasting/README.md)** - Explore third-party library integration
- 📖 **[Advanced Ensembles Guide](../../docs/advanced-ensembles.md)** - Deep dive into ensemble strategies

## Troubleshooting

### Common Issues

**Issue**: Ensemble training takes too long
```
Solution: Enable parallel training and reduce individual model complexity
Check: models/ensemble_fraud_detector.py training configuration
```

**Issue**: Real-time predictions exceed latency targets
```
Solution: Enable prediction caching and model optimization
Check: Performance tuning configuration in ensemble setup
```

**Issue**: Ensemble accuracy lower than individual models
```
Solution: Review voting strategy and model weights
Check: Correlation between individual model predictions
```

**Issue**: Memory usage too high during concurrent predictions
```
Solution: Adjust batch size and enable model sharing
Check: Resource allocation in deployment configuration
```

## Support

- **Demo-specific issues**: [GitHub Issues](https://github.com/intersystems/integratedml-demos/issues) with `demo:fraud-detection` label
- **Performance optimization**: [Performance Tuning Guide](../../docs/performance-tuning.md)
- **Ensemble patterns**: [Ensemble Best Practices](../../docs/ensemble-patterns.md)
- **Real-time integration**: [Streaming ML Guide](../../docs/streaming-ml.md)