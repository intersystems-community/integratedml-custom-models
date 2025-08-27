-- Credit Risk Model Creation for IRIS IntegratedML
-- Real implementation using actual IRIS database and IntegratedML

-- ========================================================================
-- Step 1: Create and Train Credit Risk Classification Model
-- ========================================================================

-- Create the main credit risk model using our loaded data
CREATE MODEL CreditRiskModel 
PREDICTING (default_risk) 
FROM CreditRisk.CustomerData
SELECT 
    age,
    income,
    credit_score,
    debt_to_income_ratio,
    employment_length,
    loan_amount,
    loan_purpose,
    home_ownership,
    annual_income,
    verification_status,
    default_risk;

-- Train the model
TRAIN MODEL CreditRiskModel;

-- ========================================================================
-- Step 2: Create Alternative Model Configurations
-- ========================================================================

-- Conservative model with different data selection
CREATE MODEL ConservativeCreditModel 
PREDICTING (default_risk)
FROM CreditRisk.CustomerData
SELECT 
    age,
    income,
    credit_score,
    employment_length,
    loan_amount,
    default_risk
WHERE credit_score IS NOT NULL 
    AND income > 0;

-- Train conservative model
TRAIN MODEL ConservativeCreditModel;

-- ========================================================================
-- Step 3: Validate Models
-- ========================================================================

-- Validate the main model
VALIDATE MODEL CreditRiskModel;

-- Validate conservative model
VALIDATE MODEL ConservativeCreditModel;

-- ========================================================================
-- Step 4: Check Model Information
-- ========================================================================

-- View model details
SELECT 
    model_name,
    model_type,
    trained,
    training_duration,
    validation_metric
FROM INFORMATION_SCHEMA.ML_MODELS
WHERE model_name IN ('CreditRiskModel', 'ConservativeCreditModel');

-- Check training metrics
SELECT 
    model_name,
    metric_name,
    metric_value
FROM INFORMATION_SCHEMA.ML_TRAINING_METRICS
WHERE model_name IN ('CreditRiskModel', 'ConservativeCreditModel')
ORDER BY model_name, metric_name;

-- ========================================================================
-- Step 5: Create Prediction View
-- ========================================================================

-- Create view for risk assessment
CREATE VIEW CreditRiskAssessment AS
SELECT 
    customer_id,
    age,
    income,
    credit_score,
    loan_amount,
    PREDICT(CreditRiskModel) AS risk_probability,
    CASE 
        WHEN PREDICT(CreditRiskModel) > 0.7 THEN 'HIGH'
        WHEN PREDICT(CreditRiskModel) > 0.3 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_category,
    CASE 
        WHEN PREDICT(CreditRiskModel) > 0.5 THEN 'REJECT'
        ELSE 'APPROVE'
    END AS recommendation
FROM CreditRisk.CustomerData;

-- ========================================================================
-- Step 6: Register Models in Model Registry
-- ========================================================================

-- Update model registry with trained models
UPDATE ModelRegistry 
SET status = 'ACTIVE',
    updated_at = CURRENT_TIMESTAMP,
    metadata = JSON('{"training_date": "' || CURRENT_TIMESTAMP || '", "accuracy": "pending"}')
WHERE model_name = 'CreditRiskClassifier';

-- Insert record for conservative model if not exists
INSERT OR IGNORE INTO ModelRegistry 
(model_name, model_type, demo_category, status, created_at, updated_at)
VALUES 
('ConservativeCreditModel', 'Classification', 'CreditRisk', 'ACTIVE', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- ========================================================================
-- Step 7: Sample Predictions
-- ========================================================================

-- Test predictions on a sample of data
SELECT 
    customer_id,
    age,
    credit_score,
    loan_amount,
    default_risk AS actual_risk,
    PREDICT(CreditRiskModel) AS predicted_risk,
    ABS(default_risk - PREDICT(CreditRiskModel)) AS prediction_error
FROM CreditRisk.CustomerData
WHERE customer_id <= 10
ORDER BY customer_id;

-- Show model performance summary
SELECT 
    'CreditRiskModel' AS model_name,
    COUNT(*) AS total_predictions,
    AVG(PREDICT(CreditRiskModel)) AS avg_predicted_risk,
    AVG(default_risk) AS avg_actual_risk,
    AVG(ABS(default_risk - PREDICT(CreditRiskModel))) AS mean_absolute_error,
    SQRT(AVG(POWER(default_risk - PREDICT(CreditRiskModel), 2))) AS rmse
FROM CreditRisk.CustomerData
WHERE default_risk IS NOT NULL;