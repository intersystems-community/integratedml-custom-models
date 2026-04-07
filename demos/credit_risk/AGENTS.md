# CREDIT RISK DEMO

## OVERVIEW
Loan default prediction demo — custom feature engineering for financial data (debt ratio, credit score, employment). Simplest demo structure; no `models/` directory exists on disk despite CLAUDE.md references.

## STRUCTURE
```
credit_risk/
├── data/generate_sample_data.py   # Sample data generator
├── scripts/data_preprocessing.py  # 831-line preprocessing pipeline
├── sql/
│   ├── create_model.sql           # CREATE MODEL syntax
│   ├── iris_create_model.sql      # IRIS-specific variant
│   ├── model_evaluation.sql       # VALIDATE MODEL + metrics
│   └── prediction_examples.sql    # PREDICT() query examples
├── notebooks/01_Credit_Risk_Complete_Demo.ipynb
├── tests/
│   ├── test_credit_risk_classifier.py
│   ├── test_data_preprocessing.py
│   └── test_integration.py
└── TECHNICAL_DOCUMENTATION.md
```

## MISSING: models/ directory
CLAUDE.md references `CustomCreditRiskClassifier` at `demos/credit_risk/models/credit_risk_classifier.py:22` — **this file does not exist**. The tests and SQL reference this class but no implementation is present.

## WHERE TO LOOK
| Task | File |
|------|------|
| SQL model creation | `sql/iris_create_model.sql` |
| Feature engineering | `scripts/data_preprocessing.py` |
| Integration test | `tests/test_integration.py` |

## ANTI-PATTERNS
- No conftest.py here — tests are standalone
- Don't reference `CustomCreditRiskClassifier` without creating `models/credit_risk_classifier.py` first
