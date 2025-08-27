# Demo 1: Credit Risk Assessment with Custom Feature Engineering

🟢 **Complexity**: Beginner-friendly  
🎯 **Focus**: Custom preprocessing and feature engineering within database context

## Business Problem

Financial institutions need to assess credit risk for loan applications quickly and accurately while maintaining data security. Traditional approaches require exporting sensitive financial data from secure databases, creating security vulnerabilities and compliance challenges.

## Solution Overview

This demo showcases how IntegratedML enables **custom feature engineering directly within the database**, eliminating data movement while providing domain-specific transformations for credit risk assessment.

### Key Benefits
- **Security**: Process sensitive financial data without export
- **Performance**: Eliminate ETL bottlenecks with database-resident processing
- **Compliance**: Maintain data governance and audit trails
- **Familiarity**: Use scikit-learn patterns with enhanced enterprise capabilities

## Technical Approach

### Dataset: German Credit Risk
- **Source**: UCI Machine Learning Repository (publicly available)
- **Records**: 1,000 credit applications with 20 features
- **Target**: Binary classification (good/bad credit risk)
- **Features**: Customer demographics, credit history, loan details, account information

### Custom Feature Engineering
- **Debt-to-Income Ratios**: Calculate financial health indicators
- **Credit Utilization Scores**: Assess credit usage patterns
- **Risk Interaction Terms**: Combine features for complex risk patterns
- **Domain-Specific Transformations**: Financial industry best practices

### Model Implementation
- **Base Model**: Enhanced Logistic Regression with custom preprocessing
- **Integration**: Simple scikit-learn wrapper (BaseEstimator + ClassifierMixin)
- **Complexity**: Single-file implementation for beginner accessibility

## Quick Start

### Prerequisites
```bash
# Ensure you're in the project root
cd ../../

# Install dependencies if not already done
pip install -r requirements.txt
```

### Run the Demo

1. **Explore the Data**:
   ```bash
   cd demos/credit_risk
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Train the Model**:
   ```bash
   python scripts/train_model.py
   ```

3. **Test IntegratedML Integration**:
   ```bash
   jupyter notebook notebooks/02_integratedml_integration.ipynb
   ```

## Expected User Experience

### SQL Integration
```sql
-- Create the model
CREATE MODEL CreditRiskModel PREDICTING (default_risk)
FROM CreditApplications 
USING CustomCreditRiskClassifier

-- Make predictions
SELECT customer_id, loan_amount,
       PREDICT(CreditRiskModel) as risk_probability,
       PREDICT(CreditRiskModel WITH 'class') as risk_decision
FROM NewApplications
```

### Python Integration
```python
from models.credit_risk_classifier import CustomCreditRiskClassifier

# Initialize with custom feature engineering
model = CustomCreditRiskClassifier(
    enable_debt_ratio=True,
    enable_interaction_terms=True,
    risk_threshold=0.7
)

# Train with standard scikit-learn interface
model.fit(X_train, y_train)

# Predict with confidence scores
predictions = model.predict_proba(X_test)
```

## Files and Structure

```
credit_risk/
├── README.md                    # This file
├── models/
│   ├── __init__.py
│   ├── credit_risk_classifier.py    # Main model implementation
│   ├── feature_engineering.py      # Custom feature transformers
│   └── risk_scoring.py             # Risk calculation utilities
├── data/
│   ├── README.md                    # Data sources and preparation
│   ├── german_credit.csv           # Sample dataset
│   └── data_generator.py           # Synthetic data creation
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis and insights
│   ├── 02_feature_engineering.ipynb # Custom feature development
│   ├── 03_model_training.ipynb     # Model development and training
│   └── 04_integratedml_integration.ipynb # SQL integration demo
├── scripts/
│   ├── train_model.py              # Automated training pipeline
│   ├── evaluate_model.py           # Performance evaluation
│   └── deploy_model.py             # IntegratedML deployment
├── sql/
│   ├── create_tables.sql           # Database schema setup
│   ├── create_model.sql            # IntegratedML model creation
│   └── prediction_examples.sql     # Sample prediction queries
└── tests/
    ├── test_model.py               # Model functionality tests
    ├── test_features.py            # Feature engineering tests
    └── test_integration.py         # IntegratedML integration tests
```

## Performance Expectations

### Accuracy Targets
- **Baseline Comparison**: Within 5% of standard scikit-learn LogisticRegression
- **Training Time**: < 2x baseline model on same dataset
- **Prediction Latency**: < 50ms for single transaction scoring

### Business Metrics
- **Setup Time**: Complete demo in < 15 minutes
- **Error Handling**: Clear, actionable error messages
- **Documentation**: All SQL examples copy-paste ready

> 📊 **Detailed Performance Metrics**: See our comprehensive [Performance Benchmarks](../../docs/performance_benchmarks.md) for complete accuracy, latency, and throughput measurements across all demos.

## Learning Objectives

By completing this demo, you will understand:

1. **Custom Feature Engineering**: How to implement domain-specific transformations within IntegratedML
2. **Security Benefits**: Processing sensitive data without export
3. **scikit-learn Integration**: Familiar patterns with enterprise enhancements
4. **SQL Integration**: Seamless model deployment and prediction in SQL
5. **Performance Optimization**: Database-resident processing advantages

## Next Steps

After completing this demo:
- 🟡 **[Demo 2: Fraud Detection](../fraud_detection/README.md)** - Learn ensemble model orchestration
- 🔴 **[Demo 3: Sales Forecasting](../sales_forecasting/README.md)** - Explore third-party library integration

### 📖 Comprehensive Documentation
- **[User Guide](../../docs/user_guide.md)** - Complete setup and usage guide
- **[Tutorial: Credit Risk](../../docs/tutorials/tutorial_01_credit_risk.md)** - Detailed step-by-step walkthrough
- **[Technical Architecture](../../docs/architecture.md)** - Deep dive into system design
- **[API Reference](../../docs/api_reference.md)** - Complete class and method documentation
- **[Deployment Guide](../../docs/deployment.md)** - Production deployment strategies

## Troubleshooting

### Common Issues

**Issue**: Model training fails with feature dimension mismatch
```
Solution: Ensure all categorical variables are properly encoded
Check: feature_engineering.py preprocessing pipeline
```

**Issue**: SQL CREATE MODEL statement fails
```
Solution: Verify IntegratedML is properly configured
Check: demos/credit_risk/sql/create_model.sql for syntax
```

**Issue**: Prediction accuracy lower than expected
```
Solution: Review feature engineering parameters
Check: Risk threshold and interaction term settings
```

## Support

- **Demo-specific issues**: [GitHub Issues](https://github.com/intersystems/integratedml-demos/issues) with `demo:credit-risk` label
- **General documentation**: [Project README](../../README.md)
- **IntegratedML questions**: [InterSystems Community](https://community.intersystems.com/)