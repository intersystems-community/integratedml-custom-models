# Tutorial 1: Credit Risk Assessment with Custom Feature Engineering

## 🎯 Tutorial Overview

Welcome to your first IntegratedML Pluggable Models tutorial! In this beginner-friendly guide, you'll learn how to build a custom credit risk assessment model with domain-specific feature engineering that runs directly in your database.

### What You'll Learn
- **Custom Feature Engineering**: Create financial domain-specific features like debt-to-income ratios
- **IntegratedML Integration**: Deploy ML models directly into database workflows
- **Secure Processing**: Keep sensitive financial data within your secure database environment
- **scikit-learn Compatibility**: Use familiar patterns with enhanced enterprise capabilities

### What You'll Build
A complete credit risk assessment system that:
- Processes loan applications securely without data export
- Generates custom financial risk features
- Predicts default probability with 95%+ baseline accuracy
- Integrates seamlessly with SQL workflows

**Estimated Time**: 30-45 minutes  
**Difficulty**: 🟢 Beginner-friendly  
**Prerequisites**: Basic Python and SQL knowledge

---

## 📋 Prerequisites & Setup

### System Requirements
- Python 3.8+
- 4GB RAM (recommended)
- 2GB free disk space

### Step 1: Environment Setup

```bash
# Navigate to the project root
cd /path/to/integratedml-demos

# Activate your virtual environment (recommended)
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies if not already done
pip install -r requirements.txt

# Verify installation
python -c "
from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier
print('✅ Credit Risk demo is ready!')
"
```

### Step 2: Navigate to Demo Directory

```bash
cd demos/credit_risk
ls -la
```

You should see:
```
├── README.md                    # Demo documentation
├── models/                      # Model implementations
├── data/                        # Sample datasets
├── notebooks/                   # Interactive tutorials
├── scripts/                     # Automation scripts
├── sql/                         # IntegratedML SQL examples
└── tests/                       # Validation tests
```

---

## 🏦 Understanding the Business Problem

### The Challenge
Financial institutions need to assess credit risk for loan applications while:
- **Maintaining Security**: Sensitive financial data cannot leave secure environments
- **Ensuring Compliance**: Full audit trails and governance required
- **Delivering Speed**: Real-time or near-real-time decisions needed
- **Improving Accuracy**: Better risk assessment than basic statistical models

### Traditional Approach vs IntegratedML
| Aspect | Traditional | IntegratedML |
|--------|-------------|--------------|
| **Data Movement** | Export to external systems | Process in-database |
| **Security** | Multiple data copies | Single secure location |
| **Latency** | ETL + processing delays | Direct database execution |
| **Compliance** | Complex audit trails | Built-in governance |
| **Integration** | Custom APIs needed | Native SQL interface |

### Our Solution
We'll build a custom credit risk classifier that:
1. **Generates Domain-Specific Features**: Financial ratios, risk scores, interaction terms
2. **Maintains Data Security**: All processing within database boundaries
3. **Provides Interpretable Results**: Clear risk factors and confidence scores
4. **Integrates with SQL**: Standard database workflows with ML predictions

---

## 📊 Data Exploration & Understanding

### Step 1: Generate Sample Data

```bash
# Create realistic credit risk dataset
python data/generate_sample_data.py

# Expected output:
# ✅ Generated 1000 credit applications
# ✅ Saved to: data/german_credit.csv
# ✅ Features: 20 financial attributes
# ✅ Target: Binary default risk (0=good, 1=bad)
```

### Step 2: Explore the Dataset

Launch the data exploration notebook:

```bash
jupyter notebook notebooks/credit_risk_demo.ipynb
```

**Or explore programmatically:**

```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('data/german_credit.csv')

print("📊 Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"Features: {data.columns.tolist()}")
print(f"Default Rate: {data['default_risk'].mean():.1%}")

# Display first few rows
print("\n🔍 Sample Data:")
print(data.head())

# Check for missing values
print("\n❓ Missing Values:")
print(data.isnull().sum().sum(), "total missing values")
```

### Step 3: Understand Key Features

The dataset includes typical financial features:

| Feature Category | Examples | Business Meaning |
|------------------|----------|------------------|
| **Account Info** | `checking_balance`, `savings_balance` | Customer liquidity |
| **Loan Details** | `amount`, `duration_months`, `purpose` | Loan characteristics |
| **Credit History** | `credit_history`, `installment_rate` | Payment behavior |
| **Demographics** | `age`, `employment_duration`, `housing` | Stability indicators |

**Key Insight**: Raw features need domain-specific transformations to capture financial risk patterns.

---

## 🔧 Custom Feature Engineering Deep Dive

This is where IntegratedML shines - we can create sophisticated financial features directly in the database environment.

### Step 1: Understand the Feature Engineering Pipeline

```python
from demos.credit_risk.models.credit_risk_classifier import CustomCreditRiskClassifier

# Initialize with all feature engineering enabled
model = CustomCreditRiskClassifier(
    enable_debt_ratio=True,           # Financial health indicators
    enable_interaction_terms=True,    # Complex relationship modeling
    enable_risk_scoring=True,         # Domain-specific risk calculation
    decision_threshold=0.6            # Conservative threshold for finance
)

print("🔧 Feature Engineering Configuration:")
print(f"Debt Ratio Features: {'✅' if model.enable_debt_ratio else '❌'}")
print(f"Interaction Terms: {'✅' if model.enable_interaction_terms else '❌'}")
print(f"Risk Scoring: {'✅' if model.enable_risk_scoring else '❌'}")
```

### Step 2: Debt-to-Income Ratio Features

These features capture financial health and capacity:

```python
# Example of custom debt ratio calculations
def calculate_debt_ratios(data):
    """Generate financial health indicators"""
    features = {}
    
    # Total debt to income ratio
    features['total_debt_to_income'] = data['amount'] / (data['employment_duration'] * 12 + 1)
    
    # Credit utilization proxy
    features['credit_utilization_ratio'] = data['amount'] / (data['savings_balance'] + 1)
    
    # Payment burden
    features['payment_to_income_ratio'] = (data['amount'] / data['duration_months']) / (data['employment_duration'] + 1)
    
    return pd.DataFrame(features)

# Demonstrate on sample data
sample_data = data.head()
debt_features = calculate_debt_ratios(sample_data)
print("💰 Generated Debt Ratio Features:")
print(debt_features)
```

### Step 3: Risk Interaction Terms

Capture complex relationships between features:

```python
def create_interaction_terms(data):
    """Generate interaction features for risk modeling"""
    interactions = {}
    
    # Age and credit amount interaction (young + high amount = higher risk)
    interactions['age_amount_interaction'] = data['age'] * data['amount'] / 1000
    
    # Duration and purpose interaction
    interactions['duration_purpose_interaction'] = data['duration_months'] * data['purpose']
    
    # Employment and credit history interaction
    interactions['employment_history_interaction'] = data['employment_duration'] * data['credit_history']
    
    return pd.DataFrame(interactions)

# Generate interaction features
interaction_features = create_interaction_terms(sample_data)
print("🔗 Generated Interaction Features:")
print(interaction_features)
```

### Step 4: Domain-Specific Risk Scoring

```python
def calculate_risk_scores(data):
    """Generate domain-specific risk indicators"""
    risk_scores = {}
    
    # Stability score (age, employment, housing)
    risk_scores['stability_score'] = (
        (data['age'] / 100) + 
        (data['employment_duration'] / 10) + 
        (data['housing'] == 'own').astype(int)
    ) / 3
    
    # Credit behavior score
    risk_scores['credit_behavior_score'] = (
        (data['credit_history'] >= 2).astype(int) +
        (data['checking_balance'] > 0).astype(int) +
        (data['savings_balance'] > 100).astype(int)
    ) / 3
    
    # Financial capacity score
    risk_scores['financial_capacity_score'] = np.minimum(
        data['savings_balance'] / data['amount'],
        1.0
    )
    
    return pd.DataFrame(risk_scores)

# Generate risk scores
risk_features = calculate_risk_scores(sample_data)
print("📊 Generated Risk Scores:")
print(risk_features)
```

---

## 🏗️ Model Training & Evaluation

### Step 1: Train the Model with Custom Features

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load full dataset
data = pd.read_csv('data/german_credit.csv')
X = data.drop('default_risk', axis=1)
y = data['default_risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training Set: {X_train.shape[0]} samples")
print(f"📊 Test Set: {X_test.shape[0]} samples")
print(f"📊 Default Rate - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

# Initialize and train the custom model
model = CustomCreditRiskClassifier(
    enable_debt_ratio=True,
    enable_interaction_terms=True,
    enable_risk_scoring=True,
    decision_threshold=0.6
)

print("\n🚂 Training Model...")
model.fit(X_train, y_train)
print("✅ Training Complete!")
```

### Step 2: Evaluate Model Performance

```python
# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Calculate accuracy metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob[:, 1])

print("📈 Model Performance:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc:.3f}")

# Detailed classification report
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Default Risk']))
```

### Step 3: Baseline Comparison

```python
# Compare with baseline scikit-learn model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create baseline model
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"\n📊 Performance Comparison:")
print(f"Custom Model:    {accuracy:.3f}")
print(f"Baseline Model:  {baseline_accuracy:.3f}")
print(f"Improvement:     {accuracy - baseline_accuracy:+.3f}")

if accuracy >= baseline_accuracy * 0.95:
    print("✅ Target Met: Within 5% of baseline!")
else:
    print("❌ Performance Gap: Consider tuning parameters")
```

### Step 4: Feature Importance Analysis

```python
# Analyze which custom features contribute most to predictions
model_info = model.get_model_info()
print("\n🔍 Model Information:")
print(f"Total Features: {model_info.get('n_features_in', 'Unknown')}")
print(f"Feature Engineering: {model_info.get('feature_engineering_enabled', 'Unknown')}")

# Get feature names if available
if hasattr(model, 'get_feature_names_out'):
    feature_names = model.get_feature_names_out()
    print(f"\n📋 Generated Features (first 10):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {i+1}. {name}")
```

---

## 🔌 IntegratedML Integration

This is where the magic happens - deploying your custom model directly into database workflows.

### Step 1: Model Serialization

```python
# Save the trained model
model_path = "models/trained_credit_risk_model.pkl"
model.save_model(model_path)
print(f"💾 Model saved to: {model_path}")

# Verify model can be loaded
loaded_model = CustomCreditRiskClassifier.load_model(model_path)
test_predictions = loaded_model.predict(X_test[:5])
print(f"✅ Model loading verified: {len(test_predictions)} predictions")
```

### Step 2: IntegratedML SQL Integration

Now let's see how this works in a database environment:

```sql
-- 1. Create the model in IntegratedML
CREATE MODEL CreditRiskModel PREDICTING (default_risk)
FROM CreditApplications 
USING CustomCreditRiskClassifier(
    enable_debt_ratio = 1,
    enable_interaction_terms = 1,
    enable_risk_scoring = 1,
    decision_threshold = 0.6
);

-- 2. Train the model (IntegratedML handles the data automatically)
TRAIN MODEL CreditRiskModel;

-- 3. Validate the model
VALIDATE MODEL CreditRiskModel;
```

### Step 3: Making Predictions in SQL

```sql
-- Predict default risk for new applications
SELECT 
    customer_id,
    loan_amount,
    duration_months,
    PREDICT(CreditRiskModel) as risk_probability,
    PREDICT(CreditRiskModel WITH 'class') as risk_decision,
    CASE 
        WHEN PREDICT(CreditRiskModel) > 0.6 THEN 'HIGH_RISK'
        WHEN PREDICT(CreditRiskModel) > 0.3 THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END as risk_category
FROM NewCreditApplications
WHERE application_date >= CURRENT_DATE - INTERVAL '1' DAY;

-- Batch scoring for risk management
SELECT 
    risk_category,
    COUNT(*) as application_count,
    AVG(loan_amount) as avg_loan_amount,
    SUM(loan_amount) as total_exposure
FROM (
    SELECT 
        customer_id,
        loan_amount,
        CASE 
            WHEN PREDICT(CreditRiskModel) > 0.6 THEN 'HIGH_RISK'
            WHEN PREDICT(CreditRiskModel) > 0.3 THEN 'MEDIUM_RISK'
            ELSE 'LOW_RISK'
        END as risk_category
    FROM CreditApplications
) risk_analysis
GROUP BY risk_category
ORDER BY risk_category;
```

### Step 4: Performance Monitoring

```sql
-- Monitor model performance over time
SELECT 
    DATE(application_date) as prediction_date,
    COUNT(*) as total_predictions,
    AVG(PREDICT(CreditRiskModel)) as avg_risk_score,
    COUNT(CASE WHEN PREDICT(CreditRiskModel) > 0.6 THEN 1 END) as high_risk_count,
    AVG(CASE WHEN PREDICT(CreditRiskModel) > 0.6 THEN loan_amount END) as avg_high_risk_amount
FROM CreditApplications
WHERE application_date >= CURRENT_DATE - INTERVAL '30' DAY
GROUP BY DATE(application_date)
ORDER BY prediction_date DESC;
```

---

## 🧪 Testing & Validation

### Step 1: Run Automated Tests

```bash
# Run the demo's test suite
python -m pytest tests/ -v

# Expected output:
# tests/test_credit_risk_classifier.py::test_model_initialization ✅
# tests/test_credit_risk_classifier.py::test_feature_engineering ✅
# tests/test_credit_risk_classifier.py::test_model_training ✅
# tests/test_credit_risk_classifier.py::test_predictions ✅
# tests/test_integration.py::test_integratedml_compatibility ✅
```

### Step 2: Validate Custom Features

```python
# Test custom feature generation
test_data = pd.DataFrame({
    'checking_balance': [1000, 500, 0],
    'savings_balance': [5000, 1000, 100],
    'amount': [10000, 5000, 2000],
    'duration_months': [24, 12, 6],
    'age': [35, 25, 45],
    'employment_duration': [5, 2, 10]
})

# Initialize model and generate features
feature_model = CustomCreditRiskClassifier()
enhanced_features = feature_model._create_features(test_data)

print("🧪 Feature Validation:")
print(f"Original features: {test_data.shape[1]}")
print(f"Enhanced features: {enhanced_features.shape[1]}")
print(f"Feature expansion: {enhanced_features.shape[1] / test_data.shape[1]:.1f}x")

# Verify no missing values in generated features
if enhanced_features.isnull().sum().sum() == 0:
    print("✅ No missing values in generated features")
else:
    print("❌ Missing values detected - check feature engineering")
```

### Step 3: Performance Validation

```python
# Measure prediction latency
import time

# Single prediction timing
start_time = time.time()
single_prediction = model.predict(X_test[:1])
single_latency = (time.time() - start_time) * 1000

# Batch prediction timing
start_time = time.time()
batch_predictions = model.predict(X_test[:100])
batch_latency = (time.time() - start_time) * 1000 / 100

print("⚡ Performance Validation:")
print(f"Single prediction: {single_latency:.1f}ms")
print(f"Batch prediction: {batch_latency:.1f}ms per sample")

# Validate against targets
if single_latency < 50:
    print("✅ Latency target met (<50ms)")
else:
    print("⚠️ Latency above target - consider optimization")
```

---

## 🎓 Key Learning Outcomes

Congratulations! You've successfully completed the Credit Risk Assessment tutorial. Here's what you've accomplished:

### ✅ Technical Skills Gained
- **Custom Feature Engineering**: Created financial domain-specific features
- **Model Integration**: Deployed ML models in database environments
- **Performance Optimization**: Achieved <50ms prediction latency
- **Security Best Practices**: Kept sensitive data within secure boundaries

### ✅ Business Value Delivered
- **Risk Assessment**: Built production-ready credit risk classifier
- **Compliance**: Maintained full audit trails and governance
- **Efficiency**: Eliminated data movement and ETL complexity
- **Scalability**: Created reusable patterns for financial ML

### ✅ IntegratedML Concepts Mastered
- **Pluggable Models**: Integrated custom logic with database workflows
- **scikit-learn Compatibility**: Used familiar patterns with enterprise features
- **SQL Integration**: Made ML predictions directly in database queries
- **Performance Monitoring**: Built observability into ML workflows

---

## 🚀 Next Steps

### Immediate Actions
1. **Experiment with Parameters**: Try different `decision_threshold` values
2. **Feature Engineering**: Add your own domain-specific features
3. **Performance Tuning**: Optimize for your specific use case

### Continue Learning
- **🟡 [Tutorial 2: Fraud Detection](tutorial_02_fraud_detection.md)** - Learn ensemble techniques and real-time processing
- **🔴 [Tutorial 3: Sales Forecasting](tutorial_03_sales_forecasting.md)** - Master third-party library integration
- **🔧 [Tutorial 4: Custom Models](tutorial_04_custom_models.md)** - Build your own pluggable models

### Production Deployment
- **[Deployment Guide](../deployment.md)** - Production deployment strategies
- **[Architecture Overview](../architecture.md)** - Deep dive into system design
- **[Performance Benchmarks](../performance_benchmarks.md)** - Detailed performance analysis

---

## 💡 Tips & Best Practices

### Feature Engineering Tips
- **Start Simple**: Begin with basic ratios, add complexity gradually
- **Domain Knowledge**: Leverage financial expertise for meaningful features
- **Validation**: Always validate feature distributions and missing values
- **Documentation**: Document feature business logic for compliance

### Performance Tips
- **Batch Processing**: Use batch predictions for bulk scoring
- **Caching**: Cache frequently used feature calculations
- **Monitoring**: Set up alerts for model performance degradation
- **Testing**: Comprehensive testing prevents production issues

### Security Considerations
- **Data Governance**: Ensure all processing meets compliance requirements
- **Access Control**: Implement proper database permissions
- **Audit Trails**: Maintain logs for all model predictions
- **Encryption**: Use database encryption for sensitive data

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Model accuracy below baseline
```python
# Solution: Check feature engineering parameters
model = CustomCreditRiskClassifier(
    enable_debt_ratio=True,      # Ensure key features enabled
    enable_interaction_terms=True,
    enable_risk_scoring=True
)
```

**Issue**: High prediction latency
```python
# Solution: Profile feature engineering pipeline
import time
start = time.time()
features = model._create_features(test_data)
print(f"Feature generation: {(time.time() - start) * 1000:.1f}ms")
```

**Issue**: IntegratedML integration errors
```sql
-- Solution: Verify model registration
SHOW MODELS WHERE name = 'CreditRiskModel';

-- Check model status
SELECT * FROM INFORMATION_SCHEMA.ML_MODELS 
WHERE MODEL_NAME = 'CreditRiskModel';
```

### Getting Help
- **Demo Issues**: Check [GitHub Issues](https://github.com/intersystems/integratedml-demos/issues)
- **IntegratedML Questions**: [InterSystems Community](https://community.intersystems.com/)
- **General Support**: [Documentation](../user_guide.md#getting-help)

---

**🎉 Tutorial Complete!** You've mastered the fundamentals of IntegratedML Pluggable Models with custom feature engineering. Ready for the next challenge? Try the Fraud Detection tutorial to learn ensemble techniques!