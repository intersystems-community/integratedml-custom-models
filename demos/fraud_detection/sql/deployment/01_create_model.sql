-- ============================================================================
-- Fraud Detection Ensemble Model Deployment
-- ============================================================================
-- This script creates and deploys the fraud detection ensemble model
-- in IRIS IntegratedML for real-time transaction scoring.
--
-- Author: IntegratedML Pluggable Models Team
-- Version: 1.0.0
-- ============================================================================

-- Create namespace for fraud detection if it doesn't exist
CREATE SCHEMA IF NOT EXISTS FraudDetection;

-- Switch to fraud detection namespace
USE FraudDetection;

-- ============================================================================
-- 1. Create Training Data Table
-- ============================================================================

-- Drop existing table if it exists
DROP TABLE IF EXISTS TransactionTrainingData;

-- Create comprehensive training data table
CREATE TABLE TransactionTrainingData (
    -- Primary identifiers
    transaction_id VARCHAR(255) PRIMARY KEY,
    customer_id VARCHAR(255) NOT NULL,
    
    -- Transaction details
    amount DECIMAL(15,2) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    transaction_type VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(255),
    merchant_category VARCHAR(100),
    
    -- Temporal features
    transaction_timestamp TIMESTAMP NOT NULL,
    transaction_hour INTEGER,
    transaction_day_of_week INTEGER,
    transaction_day_of_month INTEGER,
    transaction_month INTEGER,
    
    -- Location features
    merchant_country CHAR(2),
    merchant_city VARCHAR(100),
    customer_country CHAR(2),
    customer_city VARCHAR(100),
    distance_from_home DECIMAL(10,2),
    
    -- Velocity features
    transactions_last_hour INTEGER DEFAULT 0,
    transactions_last_day INTEGER DEFAULT 0,
    amount_last_hour DECIMAL(15,2) DEFAULT 0,
    amount_last_day DECIMAL(15,2) DEFAULT 0,
    
    -- Behavioral features
    avg_transaction_amount DECIMAL(15,2),
    transaction_frequency DECIMAL(8,4),
    preferred_merchants TEXT,
    unusual_time_flag INTEGER DEFAULT 0,
    unusual_location_flag INTEGER DEFAULT 0,
    
    -- Risk indicators
    high_risk_merchant INTEGER DEFAULT 0,
    international_transaction INTEGER DEFAULT 0,
    weekend_transaction INTEGER DEFAULT 0,
    late_night_transaction INTEGER DEFAULT 0,
    
    -- Customer profile features
    customer_age_days INTEGER,
    customer_tier VARCHAR(20),
    account_balance DECIMAL(15,2),
    credit_limit DECIMAL(15,2),
    
    -- Anomaly scores
    anomaly_score DECIMAL(8,6),
    behavioral_score DECIMAL(8,6),
    rule_based_score DECIMAL(8,6),
    
    -- Target variable
    is_fraud INTEGER NOT NULL,
    fraud_type VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_customer_id ON TransactionTrainingData(customer_id);
CREATE INDEX idx_transaction_timestamp ON TransactionTrainingData(transaction_timestamp);
CREATE INDEX idx_merchant_id ON TransactionTrainingData(merchant_id);
CREATE INDEX idx_is_fraud ON TransactionTrainingData(is_fraud);
CREATE INDEX idx_amount ON TransactionTrainingData(amount);

-- ============================================================================
-- 2. Create Feature Engineering Views
-- ============================================================================

-- Real-time feature view for model predictions
CREATE OR REPLACE VIEW TransactionFeatures AS
SELECT 
    transaction_id,
    customer_id,
    
    -- Basic transaction features
    amount,
    transaction_type,
    merchant_category,
    transaction_hour,
    transaction_day_of_week,
    
    -- Risk indicators
    CASE WHEN amount > 1000 THEN 1 ELSE 0 END AS high_amount_flag,
    CASE WHEN transaction_hour < 6 OR transaction_hour > 22 THEN 1 ELSE 0 END AS unusual_time_flag,
    CASE WHEN merchant_country != customer_country THEN 1 ELSE 0 END AS international_flag,
    
    -- Velocity features (calculated in real-time)
    transactions_last_hour,
    transactions_last_day,
    amount_last_hour,
    amount_last_day,
    
    -- Normalized features
    amount / NULLIF(avg_transaction_amount, 0) AS amount_ratio,
    CASE WHEN transaction_frequency > 0 THEN 1/transaction_frequency ELSE 0 END AS frequency_score,
    
    -- Anomaly scores
    anomaly_score,
    behavioral_score,
    rule_based_score,
    
    -- Composite risk score
    (anomaly_score * 0.3 + behavioral_score * 0.3 + rule_based_score * 0.4) AS composite_risk_score
    
FROM TransactionTrainingData;

-- ============================================================================
-- 3. Create Ensemble Fraud Detection Model
-- ============================================================================

-- Create the main ensemble model using custom EnsembleFraudDetector
CREATE MODEL FraudDetectionEnsemble
PREDICTING (is_fraud)
FROM TransactionFeatures
USING {
    "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
    "model_name": "EnsembleFraudDetector",
    "isc_models_disabled": 1,
    "user_params": {
        "enable_neural": 1,
        "enable_rules": 1,
        "enable_anomaly": 1,
        "enable_behavioral": 1,
        "neural_weight": 0.4,
        "rules_weight": 0.2,
        "anomaly_weight": 0.2,
        "behavioral_weight": 0.2,
        "decision_threshold": 0.5,
        "high_risk_threshold": 0.8
    }
};

-- ============================================================================
-- 4. Create Sub-Models for Ensemble Components
-- ============================================================================

-- Rule-based model for business rules (used internally by ensemble)
CREATE MODEL RuleBasedFraudDetector PREDICTING (is_fraud) FROM (
    SELECT
        transaction_id,
        high_amount_flag,
        unusual_time_flag,
        international_flag,
        high_risk_merchant,
        weekend_transaction,
        late_night_transaction,
        is_fraud
    FROM TransactionTrainingData
);

-- Anomaly detection model (used internally by ensemble)
CREATE MODEL AnomalyFraudDetector PREDICTING (is_fraud) FROM (
    SELECT
        transaction_id,
        amount,
        distance_from_home,
        anomaly_score,
        amount_ratio,
        frequency_score,
        is_fraud
    FROM TransactionFeatures
);

-- Behavioral analysis model (used internally by ensemble)
CREATE MODEL BehavioralFraudDetector PREDICTING (is_fraud) FROM (
    SELECT
        transaction_id,
        transactions_last_hour,
        transactions_last_day,
        amount_last_hour,
        amount_last_day,
        behavioral_score,
        unusual_time_flag,
        unusual_location_flag,
        is_fraud
    FROM TransactionFeatures
);

-- Neural network model for complex patterns (used internally by ensemble)
CREATE MODEL NeuralFraudDetector PREDICTING (is_fraud) FROM TransactionFeatures;

-- ============================================================================
-- 5. Create Real-time Prediction Functions
-- ============================================================================

-- Function for real-time fraud scoring
CREATE OR REPLACE FUNCTION GetFraudScore(
    p_transaction_id VARCHAR(255)
) RETURNS TABLE (
    transaction_id VARCHAR(255),
    fraud_probability DECIMAL(8,6),
    risk_level VARCHAR(20),
    ensemble_score DECIMAL(8,6),
    rule_score DECIMAL(8,6),
    anomaly_score DECIMAL(8,6),
    behavioral_score DECIMAL(8,6),
    neural_score DECIMAL(8,6),
    prediction_timestamp TIMESTAMP
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tf.transaction_id,
        
        -- Main ensemble prediction
        PREDICT(FraudDetectionEnsemble) AS fraud_probability,
        
        -- Risk level classification
        CASE 
            WHEN PREDICT(FraudDetectionEnsemble) >= 0.8 THEN 'CRITICAL'
            WHEN PREDICT(FraudDetectionEnsemble) >= 0.6 THEN 'HIGH'
            WHEN PREDICT(FraudDetectionEnsemble) >= 0.3 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS risk_level,
        
        -- Individual model scores
        PREDICT(FraudDetectionEnsemble) AS ensemble_score,
        PREDICT(RuleBasedFraudDetector) AS rule_score,
        PREDICT(AnomalyFraudDetector) AS anomaly_score,
        PREDICT(BehavioralFraudDetector) AS behavioral_score,
        PREDICT(NeuralFraudDetector) AS neural_score,
        
        CURRENT_TIMESTAMP AS prediction_timestamp
        
    FROM TransactionFeatures tf
    WHERE tf.transaction_id = p_transaction_id;
END;
$$;

-- ============================================================================
-- 6. Create Batch Prediction Procedures
-- ============================================================================

-- Procedure for batch fraud detection
CREATE OR REPLACE PROCEDURE BatchFraudDetection(
    p_start_date DATE,
    p_end_date DATE,
    p_batch_size INTEGER DEFAULT 1000
)
LANGUAGE SQL
AS $$
DECLARE
    batch_count INTEGER := 0;
    total_processed INTEGER := 0;
BEGIN
    -- Create temporary table for batch results
    CREATE TEMPORARY TABLE IF NOT EXISTS BatchFraudResults (
        transaction_id VARCHAR(255),
        fraud_probability DECIMAL(8,6),
        risk_level VARCHAR(20),
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Process transactions in batches
    FOR batch_record IN (
        SELECT transaction_id
        FROM TransactionTrainingData 
        WHERE DATE(transaction_timestamp) BETWEEN p_start_date AND p_end_date
        ORDER BY transaction_timestamp
        LIMIT p_batch_size OFFSET (batch_count * p_batch_size)
    ) LOOP
        
        -- Insert batch predictions
        INSERT INTO BatchFraudResults (transaction_id, fraud_probability, risk_level)
        SELECT 
            transaction_id,
            fraud_probability,
            risk_level
        FROM GetFraudScore(batch_record.transaction_id);
        
        total_processed := total_processed + 1;
        
        -- Commit every batch
        IF total_processed % p_batch_size = 0 THEN
            COMMIT;
            batch_count := batch_count + 1;
        END IF;
        
    END LOOP;
    
    -- Final commit
    COMMIT;
    
    -- Return summary
    RAISE NOTICE 'Batch fraud detection completed. Processed % transactions in % batches.', 
                 total_processed, batch_count + 1;
END;
$$;

-- ============================================================================
-- 7. Create Performance Monitoring Tables
-- ============================================================================

-- Model performance tracking
CREATE TABLE IF NOT EXISTS ModelPerformance (
    performance_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    evaluation_date DATE NOT NULL,
    accuracy DECIMAL(8,6),
    precision_score DECIMAL(8,6),
    recall SCORE DECIMAL(8,6),
    f1_score DECIMAL(8,6),
    auc_score DECIMAL(8,6),
    true_positives INTEGER,
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    total_predictions INTEGER,
    average_latency_ms DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction logging for audit trail
CREATE TABLE IF NOT EXISTS PredictionLog (
    log_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    customer_id VARCHAR(255) NOT NULL,
    fraud_probability DECIMAL(8,6) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    model_version VARCHAR(50),
    prediction_latency_ms DECIMAL(10,4),
    actual_fraud INTEGER,
    prediction_correct INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance monitoring
CREATE INDEX idx_model_performance_date ON ModelPerformance(evaluation_date);
CREATE INDEX idx_prediction_log_date ON PredictionLog(created_at);
CREATE INDEX idx_prediction_log_customer ON PredictionLog(customer_id);

-- ============================================================================
-- 8. Create Model Training Procedures
-- ============================================================================

-- Procedure to retrain models with new data
CREATE OR REPLACE PROCEDURE RetrainFraudModels(
    p_training_start_date DATE,
    p_training_end_date DATE
)
LANGUAGE SQL
AS $$
BEGIN
    -- Log retraining start
    RAISE NOTICE 'Starting model retraining for data from % to %', 
                 p_training_start_date, p_training_end_date;
    
    -- Retrain ensemble model with custom EnsembleFraudDetector
    DROP MODEL IF EXISTS FraudDetectionEnsemble_New;
    CREATE MODEL FraudDetectionEnsemble_New
    PREDICTING (is_fraud)
    FROM TransactionFeatures
    USING {
        "path_to_classifiers": "/opt/iris/mgr/python/custom_models/classifiers",
        "model_name": "EnsembleFraudDetector",
        "isc_models_disabled": 1,
        "user_params": {
            "enable_neural": 1,
            "enable_rules": 1,
            "enable_anomaly": 1,
            "enable_behavioral": 1,
            "neural_weight": 0.4,
            "rules_weight": 0.2,
            "anomaly_weight": 0.2,
            "behavioral_weight": 0.2,
            "decision_threshold": 0.5,
            "high_risk_threshold": 0.8
        }
    }
    WHERE DATE(transaction_timestamp) BETWEEN p_training_start_date AND p_training_end_date;
    
    -- Retrain sub-models (used internally by ensemble)
    DROP MODEL IF EXISTS RuleBasedFraudDetector_New;
    CREATE MODEL RuleBasedFraudDetector_New PREDICTING (is_fraud) FROM (
        SELECT
            transaction_id,
            high_amount_flag,
            unusual_time_flag,
            international_flag,
            high_risk_merchant,
            weekend_transaction,
            late_night_transaction,
            is_fraud
        FROM TransactionTrainingData
        WHERE DATE(transaction_timestamp) BETWEEN p_training_start_date AND p_training_end_date
    );
    
    -- Validate new models before deployment
    -- (Additional validation logic would go here)
    
    -- Deploy new models (replace old ones)
    DROP MODEL IF EXISTS FraudDetectionEnsemble_Old;
    ALTER MODEL FraudDetectionEnsemble RENAME TO FraudDetectionEnsemble_Old;
    ALTER MODEL FraudDetectionEnsemble_New RENAME TO FraudDetectionEnsemble;
    
    -- Log retraining completion
    RAISE NOTICE 'Model retraining completed successfully';
    
    COMMIT;
END;
$$;

-- ============================================================================
-- 9. Grant Permissions
-- ============================================================================

-- Grant necessary permissions for application users
GRANT SELECT ON TransactionTrainingData TO fraud_detection_app;
GRANT SELECT ON TransactionFeatures TO fraud_detection_app;
GRANT EXECUTE ON FUNCTION GetFraudScore TO fraud_detection_app;
GRANT INSERT ON PredictionLog TO fraud_detection_app;
GRANT SELECT ON ModelPerformance TO fraud_detection_app;

-- Grant permissions for admin users
GRANT ALL ON TransactionTrainingData TO fraud_detection_admin;
GRANT ALL ON ModelPerformance TO fraud_detection_admin;
GRANT ALL ON PredictionLog TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE BatchFraudDetection TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE RetrainFraudModels TO fraud_detection_admin;

-- ============================================================================
-- 10. Initialize System
-- ============================================================================

-- Create initial performance baseline
INSERT INTO ModelPerformance (
    model_name, 
    evaluation_date, 
    accuracy, 
    precision_score, 
    recall_score, 
    f1_score,
    total_predictions
) VALUES 
('FraudDetectionEnsemble', CURRENT_DATE, 0.0, 0.0, 0.0, 0.0, 0);

-- Log deployment completion
INSERT INTO PredictionLog (
    transaction_id,
    customer_id, 
    fraud_probability,
    risk_level,
    model_version,
    created_at
) VALUES (
    'SYSTEM_INIT',
    'SYSTEM',
    0.0,
    'SYSTEM',
    '1.0.0',
    CURRENT_TIMESTAMP
);

-- Display deployment summary
SELECT 
    'Fraud Detection Models Deployed Successfully' AS status,
    COUNT(*) AS total_models
FROM INFORMATION_SCHEMA.ML_MODELS 
WHERE MODEL_NAME LIKE '%Fraud%';

COMMIT;

-- ============================================================================
-- End of Model Deployment Script
-- ============================================================================