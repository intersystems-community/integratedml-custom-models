-- Create Credit Risk Assessment Model in IntegratedML
-- Demo 1: Custom Feature Engineering for Financial Risk Assessment

-- ========================================================================
-- Step 1: Create and populate the training data table
-- ========================================================================

-- Create the main credit applications table
CREATE TABLE CreditApplications (
    application_id INT PRIMARY KEY,
    customer_id VARCHAR(50),
    application_date DATE,
    
    -- Demographic information
    age INT,
    gender VARCHAR(10),
    
    -- Employment information
    employment_duration INT,
    employment_status VARCHAR(50),
    job VARCHAR(50),
    monthly_income DECIMAL(10,2),
    
    -- Housing information
    housing VARCHAR(20),
    residence_duration INT,
    
    -- Credit request details
    credit_amount DECIMAL(10,2),
    duration INT,
    purpose VARCHAR(50),
    
    -- Financial information
    existing_credits INT,
    savings_status VARCHAR(20),
    checking_status VARCHAR(20),
    
    -- Credit history
    credit_history VARCHAR(50),
    num_dependents INT,
    telephone VARCHAR(10),
    foreign_worker VARCHAR(5),
    
    -- Target variable for training
    default_risk INT,  -- 0 = good credit, 1 = bad credit
    
    -- Audit fields
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster training
CREATE INDEX idx_credit_applications_date ON CreditApplications(application_date);
CREATE INDEX idx_credit_applications_risk ON CreditApplications(default_risk);

-- Create table for new applications (without target variable)
CREATE TABLE NewCreditApplications (
    application_id INT PRIMARY KEY,
    customer_id VARCHAR(50),
    application_date DATE,
    application_status VARCHAR(20) DEFAULT 'PENDING',
    
    -- Same structure as training table but without default_risk
    age INT,
    gender VARCHAR(10),
    employment_duration INT,
    employment_status VARCHAR(50),
    job VARCHAR(50),
    monthly_income DECIMAL(10,2),
    housing VARCHAR(20),
    residence_duration INT,
    credit_amount DECIMAL(10,2),
    duration INT,
    purpose VARCHAR(50),
    existing_credits INT,
    savings_status VARCHAR(20),
    checking_status VARCHAR(20),
    credit_history VARCHAR(50),
    num_dependents INT,
    telephone VARCHAR(10),
    foreign_worker VARCHAR(5),
    
    -- Fields for tracking predictions
    predicted_risk_score DECIMAL(5,4),
    predicted_risk_class INT,
    prediction_date TIMESTAMP,
    final_decision VARCHAR(20),
    
    -- Audit fields
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================================================
-- Step 2: Load sample data (this would typically be done via data import)
-- ========================================================================

-- Note: In a real implementation, you would load data from the generated CSV files
-- COPY CreditApplications FROM 'demos/credit_risk/data/credit_risk_train.csv'
-- WITH DELIMITER ',' CSV HEADER;

-- ========================================================================
-- Step 3: Create IntegratedML models with different configurations
-- ========================================================================

-- Main credit risk model with full feature engineering
CREATE MODEL CreditRiskModel PREDICTING (default_risk)
FROM CreditApplications
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/credit_risk/iris_models",
    ,
    
    "iscmodelsdisabled": 1,
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_interaction_terms": 1,
        "enable_risk_scoring": 1,
        "decision_threshold": 0.6
    }
};

-- Conservative model with stricter threshold for high-stakes decisions
CREATE MODEL ConservativeCreditModel PREDICTING (default_risk)
FROM CreditApplications
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/credit_risk/iris_models",
    ,
    
    "iscmodelsdisabled": 1,
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_interaction_terms": 0,
        "enable_risk_scoring": 1,
        "decision_threshold": 0.4
    }
};

-- Lightweight model without interaction terms for faster scoring
CREATE MODEL FastCreditModel PREDICTING (default_risk)
FROM CreditApplications
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/credit_risk/iris_models",
    ,
    
    "iscmodelsdisabled": 1,
    "user_params": {
        "enable_debt_ratio": 1,
        "enable_interaction_terms": 0,
        "enable_risk_scoring": 0,
        "decision_threshold": 0.5
    }
};

-- ========================================================================
-- Step 4: Train the models
-- ========================================================================

-- Train the main model
TRAIN MODEL CreditRiskModel;

-- Train alternative models
TRAIN MODEL ConservativeCreditModel;
TRAIN MODEL FastCreditModel;

-- ========================================================================
-- Step 5: Validate models and check training results
-- ========================================================================

-- Validate the main model
VALIDATE MODEL CreditRiskModel;

-- Check model training metrics
SELECT
    model_name,
    training_start,
    training_end,
    training_samples,
    validation_accuracy,
    validation_auc
FROM INFORMATION_SCHEMA.ML_TRAINING_RUNS
WHERE model_name IN ('CreditRiskModel', 'ConservativeCreditModel', 'FastCreditModel')
ORDER BY training_end DESC;

-- Show detailed model information
SELECT
    model_name,
    provider_name,
    model_type,
    target_column,
    training_table,
    model_parameters,
    created_date
FROM INFORMATION_SCHEMA.ML_MODELS
WHERE model_name LIKE '%CreditModel'
ORDER BY created_date;

-- ========================================================================
-- Step 6: Create views for easier model usage
-- ========================================================================

-- View that combines applications with risk scoring
CREATE VIEW ApplicationsWithRisk AS
SELECT
    a.*,
    PREDICT(CreditRiskModel USING a.*) as predicted_risk_score,
    PREDICT(CreditRiskModel WITH 'class' USING a.*) as predicted_risk_class,
    CASE
        WHEN PREDICT(CreditRiskModel USING a.*) >= 0.7 THEN 'HIGH'
        WHEN PREDICT(CreditRiskModel USING a.*) >= 0.3 THEN 'MEDIUM'
        ELSE 'LOW'
    END as risk_category
FROM NewCreditApplications a;

-- View for risk monitoring dashboard
CREATE VIEW RiskMonitoringDashboard AS
SELECT
    DATE(application_date) as application_date,
    COUNT(*) as total_applications,
    AVG(predicted_risk_score) as avg_risk_score,
    SUM(CASE WHEN predicted_risk_class = 1 THEN 1 ELSE 0 END) as high_risk_count,
    SUM(CASE WHEN predicted_risk_class = 0 THEN 1 ELSE 0 END) as low_risk_count,
    AVG(credit_amount) as avg_credit_amount
FROM ApplicationsWithRisk
GROUP BY DATE(application_date)
ORDER BY application_date DESC;