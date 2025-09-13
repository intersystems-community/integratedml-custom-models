# IntegratedML Custom Models Feature - Product Requirements Document

## Executive Summary

IntegratedML now supports **custom Python model integration**, enabling data scientists and developers to bring their own machine learning models directly into InterSystems IRIS SQL workflows. This groundbreaking feature allows custom Python preprocessing, feature engineering, and model training code to be executed within SQL commands like `CREATE MODEL` and `SELECT ... PREDICT()`.

## Core Value Proposition

### Before IntegratedML Custom Models
```python
# Traditional approach - data movement required
data = fetch_from_database()
processed_data = custom_preprocessing(data)
model = train_custom_model(processed_data)
predictions = model.predict(new_data)
write_to_database(predictions)
```

### With IntegratedML Custom Models
```sql
-- Everything happens in-database!
CREATE MODEL FraudDetectionModel
PREDICTING (is_fraud)
FROM Transactions
USING "demos.fraud_detection.models.EnsembleFraudDetector";

-- Real-time predictions without data movement
SELECT transaction_id, amount,
       PREDICT(FraudDetectionModel) as fraud_risk
FROM LiveTransactions
WHERE amount > 1000;
```

## Key Features

### 1. Seamless SQL Integration
- Use familiar SQL syntax to train and deploy custom ML models
- No need to export data or manage separate ML infrastructure
- Models execute where the data lives

### 2. Python Flexibility
- Bring any scikit-learn compatible model
- Custom preprocessing and feature engineering
- Support for ensemble methods and complex architectures
- Integration with popular libraries (TensorFlow, LightGBM, Prophet)

### 3. Production-Ready
- Models persist in the database
- Automatic versioning and lifecycle management
- Built-in security and access controls
- Scalable in-database execution

## How It Works

### Step 1: Define Your Custom Model
```python
from shared.models.base import IntegratedMLBaseModel

class CustomCreditRiskClassifier(IntegratedMLBaseModel):
    def fit(self, X, y):
        # Custom feature engineering
        X_engineered = self._engineer_features(X)
        # Train your model
        self.model = LogisticRegression()
        self.model.fit(X_engineered, y)
        return self

    def predict(self, X):
        X_engineered = self._engineer_features(X)
        return self.model.predict(X_engineered)
```

### Step 2: Train Using SQL
```sql
CREATE MODEL CreditRiskModel
PREDICTING (default_risk)
FROM CreditApplications
USING "demos.credit_risk.models.CustomCreditRiskClassifier"
WITH (enable_debt_ratio=true, decision_threshold=0.7);
```

### Step 3: Validate Model Performance
```sql
VALIDATE MODEL CreditRiskModel
FROM TestApplications;
```

### Step 4: Make Predictions
```sql
SELECT
    customer_id,
    credit_amount,
    PREDICT(CreditRiskModel) as risk_prediction,
    PREDICT(CreditRiskModel PROBABILITY) as risk_probability
FROM NewApplications;
```

## Demo Showcase

This repository demonstrates four real-world use cases:

### 1. Credit Risk Assessment
**Problem**: Banks need to assess credit risk while keeping sensitive financial data secure.

**Solution**: Custom feature engineering (debt-to-income ratios, stability scores) executed in-database.

**Key Features**:
- Domain-specific financial calculations
- Risk scoring with explanations
- Compliance-friendly in-database processing

### 2. Real-time Fraud Detection
**Problem**: Payment processors need sub-100ms fraud detection without data movement latency.

**Solution**: Ensemble model combining neural networks, rules, and behavioral analysis.

**Key Features**:
- Multiple model voting strategies
- Real-time feature calculation
- Configurable confidence thresholds

### 3. Sales Forecasting
**Problem**: Retailers need accurate forecasts combining time series and ML approaches.

**Solution**: Hybrid model integrating Prophet (trending) with LightGBM (pattern learning).

**Key Features**:
- Third-party library integration
- Confidence interval generation
- Seasonal decomposition

### 4. DNA Sequence Similarity
**Problem**: Genomics researchers need specialized sequence analysis algorithms.

**Solution**: Custom similarity metrics (Levenshtein distance, k-mer analysis) in SQL.

**Key Features**:
- Bioinformatics algorithms
- Specialized distance calculations
- Optimized sequence processing

## Technical Architecture

### Model Lifecycle
1. **Development**: Create Python model inheriting from `IntegratedMLBaseModel`
2. **Registration**: Model path specified in SQL `USING` clause
3. **Training**: SQL `CREATE MODEL` triggers Python fit() method
4. **Validation**: SQL `VALIDATE MODEL` evaluates performance metrics
5. **Persistence**: Trained model stored in IRIS
6. **Inference**: SQL `PREDICT()` calls Python predict() method

### Integration Points
- **Parameter Passing**: SQL `WITH` clause → Python `__init__` parameters
- **Data Transfer**: IRIS tables → Pandas DataFrames
- **Model Storage**: Python pickle → IRIS model repository
- **Error Handling**: Python exceptions → SQL error messages

## Benefits for Different Personas

### For Data Scientists
- Use familiar Python ML libraries
- No infrastructure management
- Focus on model development, not deployment

### For SQL Developers
- Access ML capabilities through SQL
- No Python knowledge required for predictions
- Consistent SQL-based workflows

### For IT/Operations
- Reduced infrastructure complexity
- Unified security model
- Simplified model governance

### For Business Users
- Faster time-to-insight
- Real-time predictions on live data
- Lower total cost of ownership

## Getting Started

### Quick Demo (5 minutes)
```bash
# Clone and setup
git clone <repo>
cd integratedml-flexible-model-integration
make setup

# Run a demo
make demo-credit
```

### Try Your Own Model (15 minutes)
1. Create a model class inheriting from `IntegratedMLBaseModel`
2. Implement `fit()` and `predict()` methods
3. Use SQL to train: `CREATE MODEL ... USING "your.model.path"`
4. Validate performance: `VALIDATE MODEL YourModel FROM TestData`
5. Make predictions: `SELECT PREDICT(YourModel) FROM YourTable`

## Success Metrics

- **Performance**: In-database execution eliminates data movement latency
- **Flexibility**: Support for any Python ML approach
- **Adoption**: Simple SQL interface for complex ML
- **Governance**: Centralized model management

## Roadmap

### Current Release
- Scikit-learn compatible models
- Basic parameter passing
- Model persistence
- SQL prediction functions

### Future Enhancements
- Model versioning and A/B testing
- Automated retraining triggers
- Model explainability APIs
- Distributed training support

## Call to Action

1. **Explore the Demos**: See how custom models solve real problems
2. **Build Your Own**: Create models for your specific use cases
3. **Share Feedback**: Help shape the future of IntegratedML
4. **Join the Community**: Share your models and learn from others

---

*IntegratedML Custom Models Feature - Where SQL Meets Machine Learning*