# IntegratedML Custom Models - Early Access Program

Deploy custom Python ML models directly within SQL queries using InterSystems IRIS.

## What This Feature Does

IntegratedML Custom Models extends the existing IntegratedML/AutoML capability by allowing you to deploy your own custom Python models directly within SQL. While IntegratedML AutoML provides automated machine learning, Custom Models gives you full control over model training, preprocessing, and predictions - all while keeping the same SQL interface.

Use this when you need:
• Custom preprocessing or feature engineering
• Domain-specific algorithms
• Third-party libraries (Prophet, LightGBM, XGBoost)
• Full control over model training logic

## Requirements

IRIS Version: 2025.2 or later (Community Edition or licensed)

IntegratedML/AutoML: Must be installed and configured in your IRIS instance. If not already installed, use:

python -m pip install --index-url https://registry.intersystems.com/pypi/simple --no-cache-dir --target /usr/irissys/mgr/python intersystems-iris-automl

Python: 3.8 or later (3.11+ recommended)

Platform: macOS (primary support), Linux or Windows (secondary support)

## Getting Started

Repository: https://github.com/intersystems-community/integratedml-custom-models

1. Read the EAP Guide: https://github.com/intersystems-community/integratedml-custom-models/blob/main/docs/EAP_GUIDE.md

2. Follow Installation Guide: https://github.com/intersystems-community/integratedml-custom-models/blob/main/docs/INSTALLATION.md

3. Check Known Issues: https://github.com/intersystems-community/integratedml-custom-models/blob/main/docs/EAP_KNOWN_ISSUES.md

Target installation time: Under 30 minutes

## What's Included

The repository includes:
• 4 complete demo applications (Credit Risk, Fraud Detection, Sales Forecasting, DNA Similarity)
• Comprehensive documentation (installation, troubleshooting, API reference)
• Base model classes for classification, regression, and ensemble models
• Docker setup for quick evaluation

## EAP Program Details

Duration: Approximately 6-8 weeks
Participants: 5 selected users
Target GA Release: IRIS 2026.1 (Q2 2026)

Your feedback will directly shape the final product. We're looking for feedback on:
• Installation experience
• Documentation clarity
• Feature completeness
• Production readiness considerations

## How to Provide Feedback

Survey (recommended): Survey links will be provided by the Data Platforms Product Team

Email: thomas.dyar@intersystems.com (response time: 1-2 business days)

GitHub Issues (if enabled): Technical bugs and feature requests

## Support During EAP

For questions or issues:
• Check the Troubleshooting Guide: https://github.com/intersystems-community/integratedml-custom-models/blob/main/docs/TROUBLESHOOTING.md
• Check the FAQ: https://github.com/intersystems-community/integratedml-custom-models/blob/main/docs/EAP_FAQ.md
• Email: thomas.dyar@intersystems.com

Target: Less than 1 support request per participant for installation

## Important Notes

• This is pre-release software for evaluation only - not for production use during EAP
• API may change based on feedback
• Full production support will be available in IRIS 2026.1 GA release
• Documentation and examples are continuously updated based on participant feedback

## Example Usage

After installation, you can create and use custom models with standard SQL commands:

CREATE MODEL CreditRiskModel
PREDICTING (default_risk)
FROM CreditApplications
USING '{"model_name": "CustomCreditRiskClassifier", "path_to_classifiers": "/path/to/models"}'

TRAIN MODEL CreditRiskModel

SELECT customer_id, PREDICT(CreditRiskModel) as risk_score
FROM NewApplications

## Complete Documentation

All documentation is available in the GitHub repository:
https://github.com/intersystems-community/integratedml-custom-models

Thank you for participating in the Early Access Program!
