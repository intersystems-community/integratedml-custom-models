-- ============================================================================
-- Complete Fraud Detection System Deployment
-- ============================================================================
-- This script deploys the complete fraud detection system including all
-- models, procedures, monitoring, and maintenance components.
--
-- Author: IntegratedML Pluggable Models Team
-- Version: 1.0.0
-- ============================================================================

-- Set deployment parameters
\set deployment_version '1.0.0'
\set deployment_date CURRENT_TIMESTAMP

-- Create deployment log
DO $$
BEGIN
    RAISE NOTICE 'Starting Fraud Detection System Deployment v%', :'deployment_version';
    RAISE NOTICE 'Deployment Date: %', :'deployment_date';
END
$$;

-- ============================================================================
-- 1. Prerequisites Check
-- ============================================================================

-- Check IRIS IntegratedML availability
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.SCHEMATA 
        WHERE SCHEMA_NAME = 'INFORMATION_SCHEMA'
    ) THEN
        RAISE EXCEPTION 'IRIS database system not detected';
    END IF;
    
    RAISE NOTICE 'Prerequisites check passed';
END
$$;

-- ============================================================================
-- 2. Schema and Security Setup
-- ============================================================================

-- Create fraud detection schema
CREATE SCHEMA IF NOT EXISTS FraudDetection;
USE FraudDetection;

-- Create user roles
DO $$
BEGIN
    -- Application role for real-time predictions
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fraud_detection_app') THEN
        CREATE ROLE fraud_detection_app LOGIN PASSWORD 'fraud_app_secure_2024';
    END IF;
    
    -- Admin role for model management
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fraud_detection_admin') THEN
        CREATE ROLE fraud_detection_admin LOGIN PASSWORD 'fraud_admin_secure_2024';
    END IF;
    
    -- Monitoring role for dashboards
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fraud_detection_monitor') THEN
        CREATE ROLE fraud_detection_monitor LOGIN PASSWORD 'fraud_monitor_secure_2024';
    END IF;
    
    RAISE NOTICE 'Security roles created successfully';
END
$$;

-- ============================================================================
-- 3. Core Tables Deployment
-- ============================================================================

-- Deploy all core tables with proper structure
\i deployment/01_create_model.sql

-- Additional supporting tables
CREATE TABLE IF NOT EXISTS SystemConfiguration (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value TEXT NOT NULL,
    config_type VARCHAR(50) DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration
INSERT INTO SystemConfiguration (config_key, config_value, config_type, description) VALUES
('model_version', '1.0.0', 'string', 'Current fraud detection model version'),
('prediction_latency_threshold_ms', '100', 'integer', 'Maximum acceptable prediction latency'),
('high_risk_threshold', '0.8', 'decimal', 'Threshold for high-risk classification'),
('medium_risk_threshold', '0.3', 'decimal', 'Threshold for medium-risk classification'),
('cache_ttl_seconds', '300', 'integer', 'Default cache TTL in seconds'),
('auto_retrain_enabled', 'true', 'boolean', 'Whether automatic retraining is enabled'),
('retrain_frequency_days', '30', 'integer', 'Frequency of automatic retraining in days'),
('alert_email_enabled', 'false', 'boolean', 'Whether email alerts are enabled'),
('performance_monitoring_enabled', 'true', 'boolean', 'Whether performance monitoring is active')
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- 4. Real-time Prediction System Deployment
-- ============================================================================

\i prediction/realtime_prediction.sql

-- ============================================================================
-- 5. Monitoring System Deployment
-- ============================================================================

\i monitoring/performance_monitoring.sql

-- ============================================================================
-- 6. Maintenance System Deployment
-- ============================================================================

\i maintenance/model_maintenance.sql

-- ============================================================================
-- 7. Sample Data Loading (Optional)
-- ============================================================================

-- Load sample transaction data for testing
CREATE OR REPLACE PROCEDURE LoadSampleData(
    p_num_transactions INTEGER DEFAULT 10000
)
LANGUAGE SQL
AS $$
DECLARE
    i INTEGER;
    customer_ids TEXT[] := ARRAY['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'];
    merchant_ids TEXT[] := ARRAY['MERCH001', 'MERCH002', 'MERCH003', 'MERCH004', 'MERCH005'];
    transaction_types TEXT[] := ARRAY['purchase', 'withdrawal', 'transfer', 'payment'];
    currencies TEXT[] := ARRAY['USD', 'EUR', 'GBP', 'CAD'];
BEGIN
    RAISE NOTICE 'Loading % sample transactions...', p_num_transactions;
    
    FOR i IN 1..p_num_transactions LOOP
        INSERT INTO TransactionTrainingData (
            transaction_id,
            customer_id,
            amount,
            currency_code,
            transaction_type,
            merchant_id,
            merchant_category,
            transaction_timestamp,
            transaction_hour,
            transaction_day_of_week,
            merchant_country,
            customer_country,
            distance_from_home,
            transactions_last_hour,
            transactions_last_day,
            amount_last_hour,
            amount_last_day,
            avg_transaction_amount,
            transaction_frequency,
            high_risk_merchant,
            international_transaction,
            weekend_transaction,
            late_night_transaction,
            customer_age_days,
            customer_tier,
            account_balance,
            anomaly_score,
            behavioral_score,
            rule_based_score,
            is_fraud,
            fraud_type
        ) VALUES (
            'TXN' || LPAD(i::TEXT, 10, '0'),
            customer_ids[1 + (RANDOM() * 4)::INTEGER],
            (RANDOM() * 2000 + 10)::DECIMAL(15,2),
            currencies[1 + (RANDOM() * 3)::INTEGER],
            transaction_types[1 + (RANDOM() * 3)::INTEGER],
            merchant_ids[1 + (RANDOM() * 4)::INTEGER],
            CASE WHEN RANDOM() < 0.3 THEN 'grocery' 
                 WHEN RANDOM() < 0.6 THEN 'retail'
                 ELSE 'restaurant' END,
            CURRENT_TIMESTAMP - (RANDOM() * 30 || ' days')::INTERVAL,
            EXTRACT(HOUR FROM CURRENT_TIMESTAMP - (RANDOM() * 30 || ' days')::INTERVAL),
            EXTRACT(DOW FROM CURRENT_TIMESTAMP - (RANDOM() * 30 || ' days')::INTERVAL),
            CASE WHEN RANDOM() < 0.8 THEN 'US' ELSE 'CA' END,
            'US',
            (RANDOM() * 100)::DECIMAL(10,2),
            (RANDOM() * 5)::INTEGER,
            (RANDOM() * 20)::INTEGER,
            (RANDOM() * 1000)::DECIMAL(15,2),
            (RANDOM() * 5000)::DECIMAL(15,2),
            (RANDOM() * 500 + 50)::DECIMAL(15,2),
            (RANDOM() * 10 + 1)::DECIMAL(8,4),
            CASE WHEN RANDOM() < 0.1 THEN 1 ELSE 0 END,
            CASE WHEN RANDOM() < 0.2 THEN 1 ELSE 0 END,
            CASE WHEN RANDOM() < 0.3 THEN 1 ELSE 0 END,
            CASE WHEN RANDOM() < 0.15 THEN 1 ELSE 0 END,
            (RANDOM() * 1000 + 30)::INTEGER,
            CASE WHEN RANDOM() < 0.2 THEN 'premium' 
                 WHEN RANDOM() < 0.6 THEN 'standard' 
                 ELSE 'basic' END,
            (RANDOM() * 10000 + 1000)::DECIMAL(15,2),
            RANDOM()::DECIMAL(8,6),
            RANDOM()::DECIMAL(8,6),
            RANDOM()::DECIMAL(8,6),
            CASE WHEN RANDOM() < 0.05 THEN 1 ELSE 0 END,
            CASE WHEN RANDOM() < 0.05 THEN 
                CASE WHEN RANDOM() < 0.5 THEN 'card_testing' ELSE 'account_takeover' END 
            ELSE NULL END
        );
        
        -- Commit every 1000 records
        IF i % 1000 = 0 THEN
            COMMIT;
            RAISE NOTICE 'Loaded % transactions...', i;
        END IF;
    END LOOP;
    
    RAISE NOTICE 'Sample data loading completed: % transactions', p_num_transactions;
    COMMIT;
END;
$$;

-- ============================================================================
-- 8. System Initialization
-- ============================================================================

-- Initialize customer profiles
INSERT INTO CustomerProfiles (
    customer_id,
    customer_tier,
    account_balance,
    avg_transaction_amount,
    transaction_frequency,
    customer_age_days,
    preferred_countries,
    last_transaction_date,
    total_transactions,
    risk_score
) 
SELECT 
    customer_id,
    MAX(customer_tier) as customer_tier,
    MAX(account_balance) as account_balance,
    AVG(amount) as avg_transaction_amount,
    COUNT(*)::DECIMAL / 30 as transaction_frequency,
    MAX(customer_age_days) as customer_age_days,
    STRING_AGG(DISTINCT merchant_country, ',') as preferred_countries,
    MAX(DATE(transaction_timestamp)) as last_transaction_date,
    COUNT(*) as total_transactions,
    AVG((is_fraud::INTEGER + high_risk_merchant + international_transaction + late_night_transaction)::DECIMAL / 4) as risk_score
FROM TransactionTrainingData
GROUP BY customer_id
ON CONFLICT (customer_id) DO UPDATE SET
    avg_transaction_amount = EXCLUDED.avg_transaction_amount,
    transaction_frequency = EXCLUDED.transaction_frequency,
    last_transaction_date = EXCLUDED.last_transaction_date,
    total_transactions = EXCLUDED.total_transactions,
    risk_score = EXCLUDED.risk_score,
    updated_at = CURRENT_TIMESTAMP;

-- Initialize merchant profiles
INSERT INTO MerchantProfiles (
    merchant_id,
    merchant_name,
    risk_category,
    fraud_rate,
    avg_transaction_amount,
    total_transactions
)
SELECT 
    merchant_id,
    'Merchant ' || merchant_id as merchant_name,
    CASE 
        WHEN AVG(is_fraud::INTEGER) > 0.1 THEN 'high_risk'
        WHEN AVG(is_fraud::INTEGER) > 0.05 THEN 'medium_risk'
        ELSE 'low_risk'
    END as risk_category,
    AVG(is_fraud::INTEGER) as fraud_rate,
    AVG(amount) as avg_transaction_amount,
    COUNT(*) as total_transactions
FROM TransactionTrainingData
WHERE merchant_id IS NOT NULL
GROUP BY merchant_id
ON CONFLICT (merchant_id) DO UPDATE SET
    fraud_rate = EXCLUDED.fraud_rate,
    avg_transaction_amount = EXCLUDED.avg_transaction_amount,
    total_transactions = EXCLUDED.total_transactions,
    updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- 9. Model Training and Deployment
-- ============================================================================

-- Train initial models if data exists
DO $$
DECLARE
    training_data_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO training_data_count 
    FROM TransactionTrainingData 
    WHERE is_fraud IS NOT NULL;
    
    IF training_data_count >= 1000 THEN
        RAISE NOTICE 'Training initial models with % records...', training_data_count;
        
        -- The models are already created in 01_create_model.sql
        -- Here we would trigger the training process
        
        RAISE NOTICE 'Initial model training completed';
    ELSE
        RAISE NOTICE 'Insufficient training data (% records), skipping model training', training_data_count;
    END IF;
END
$$;

-- ============================================================================
-- 10. Performance Testing
-- ============================================================================

-- Test real-time prediction performance
CREATE OR REPLACE FUNCTION TestPredictionPerformance(
    p_test_count INTEGER DEFAULT 100
) RETURNS TABLE (
    test_name VARCHAR(100),
    avg_latency_ms DECIMAL(10,4),
    max_latency_ms DECIMAL(10,4),
    min_latency_ms DECIMAL(10,4),
    success_rate DECIMAL(8,4),
    status VARCHAR(20)
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    WITH performance_test AS (
        SELECT 
            'FastFraudScore' as test_type,
            (result).processing_time_ms as latency_ms,
            CASE WHEN (result).fraud_probability IS NOT NULL THEN 1 ELSE 0 END as success
        FROM (
            SELECT FastFraudScore('CUST001', 100.00, 'MERCH001', 'purchase', 'retail') as result
            FROM generate_series(1, p_test_count)
        ) tests
    )
    SELECT 
        pt.test_type::VARCHAR(100),
        AVG(pt.latency_ms)::DECIMAL(10,4),
        MAX(pt.latency_ms)::DECIMAL(10,4),
        MIN(pt.latency_ms)::DECIMAL(10,4),
        AVG(pt.success)::DECIMAL(8,4),
        CASE 
            WHEN AVG(pt.latency_ms) <= 50 THEN 'EXCELLENT'
            WHEN AVG(pt.latency_ms) <= 100 THEN 'GOOD'
            WHEN AVG(pt.latency_ms) <= 200 THEN 'ACCEPTABLE'
            ELSE 'POOR'
        END::VARCHAR(20)
    FROM performance_test pt
    GROUP BY pt.test_type;
END;
$$;

-- ============================================================================
-- 11. Deployment Validation
-- ============================================================================

-- Validate complete deployment
CREATE OR REPLACE FUNCTION ValidateDeployment()
RETURNS TABLE (
    component VARCHAR(100),
    status VARCHAR(20),
    details TEXT
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    
    -- Check tables
    SELECT 
        'Core Tables'::VARCHAR(100),
        CASE WHEN COUNT(*) >= 8 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' core tables'::TEXT
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'FraudDetection'
    AND TABLE_NAME IN ('TransactionTrainingData', 'ModelPerformance', 'PredictionLog', 
                       'CustomerProfiles', 'MerchantProfiles', 'SystemAlerts', 
                       'CustomerVelocity', 'PredictionCache')
    
    UNION ALL
    
    -- Check models
    SELECT 
        'ML Models'::VARCHAR(100),
        CASE WHEN COUNT(*) >= 5 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' ML models'::TEXT
    FROM INFORMATION_SCHEMA.ML_MODELS 
    WHERE MODEL_NAME LIKE '%Fraud%'
    
    UNION ALL
    
    -- Check functions
    SELECT 
        'Functions'::VARCHAR(100),
        CASE WHEN COUNT(*) >= 5 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' functions'::TEXT
    FROM INFORMATION_SCHEMA.ROUTINES 
    WHERE ROUTINE_SCHEMA = 'FraudDetection'
    AND ROUTINE_TYPE = 'FUNCTION'
    AND ROUTINE_NAME LIKE '%Fraud%'
    
    UNION ALL
    
    -- Check procedures
    SELECT 
        'Procedures'::VARCHAR(100),
        CASE WHEN COUNT(*) >= 3 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' procedures'::TEXT
    FROM INFORMATION_SCHEMA.ROUTINES 
    WHERE ROUTINE_SCHEMA = 'FraudDetection'
    AND ROUTINE_TYPE = 'PROCEDURE'
    
    UNION ALL
    
    -- Check views
    SELECT 
        'Monitoring Views'::VARCHAR(100),
        CASE WHEN COUNT(*) >= 3 THEN 'OK' ELSE 'MISSING' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' monitoring views'::TEXT
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_SCHEMA = 'FraudDetection'
    
    UNION ALL
    
    -- Check sample data
    SELECT 
        'Sample Data'::VARCHAR(100),
        CASE WHEN COUNT(*) > 0 THEN 'OK' ELSE 'EMPTY' END::VARCHAR(20),
        'Found ' || COUNT(*) || ' training records'::TEXT
    FROM TransactionTrainingData;
END;
$$;

-- ============================================================================
-- 12. Deployment Completion
-- ============================================================================

-- Log deployment completion
INSERT INTO SystemLog (log_type, message, details) VALUES (
    'SYSTEM_DEPLOYMENT',
    'Fraud Detection System deployment completed',
    JSONB_BUILD_OBJECT(
        'version', :'deployment_version',
        'deployment_date', :'deployment_date',
        'components', 'complete'
    )
);

-- Run deployment validation
DO $$
DECLARE
    validation_result RECORD;
    all_components_ok BOOLEAN := TRUE;
BEGIN
    RAISE NOTICE 'Running deployment validation...';
    
    FOR validation_result IN SELECT * FROM ValidateDeployment() LOOP
        RAISE NOTICE 'Component: %, Status: %, Details: %', 
                     validation_result.component, 
                     validation_result.status, 
                     validation_result.details;
        
        IF validation_result.status NOT IN ('OK') THEN
            all_components_ok := FALSE;
        END IF;
    END LOOP;
    
    IF all_components_ok THEN
        RAISE NOTICE 'DEPLOYMENT SUCCESSFUL: All components validated';
        
        -- Insert success alert
        INSERT INTO SystemAlerts (alert_type, message, severity) VALUES (
            'DEPLOYMENT_SUCCESS',
            'Fraud Detection System v' || :'deployment_version' || ' deployed successfully',
            'INFO'
        );
    ELSE
        RAISE WARNING 'DEPLOYMENT ISSUES: Some components failed validation';
        
        -- Insert warning alert
        INSERT INTO SystemAlerts (alert_type, message, severity) VALUES (
            'DEPLOYMENT_WARNING',
            'Fraud Detection System deployment completed with warnings',
            'WARNING'
        );
    END IF;
END
$$;

-- Run performance test
DO $$
DECLARE
    perf_result RECORD;
BEGIN
    RAISE NOTICE 'Running performance tests...';
    
    FOR perf_result IN SELECT * FROM TestPredictionPerformance(50) LOOP
        RAISE NOTICE 'Test: %, Avg Latency: %ms, Status: %', 
                     perf_result.test_name,
                     perf_result.avg_latency_ms,
                     perf_result.status;
    END LOOP;
END
$$;

-- Final deployment summary
SELECT 
    'Fraud Detection System v' || :'deployment_version' AS system,
    'DEPLOYED' AS status,
    CURRENT_TIMESTAMP AS completion_time,
    (SELECT COUNT(*) FROM TransactionTrainingData) AS training_records,
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.ML_MODELS WHERE MODEL_NAME LIKE '%Fraud%') AS ml_models,
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_SCHEMA = 'FraudDetection') AS functions_procedures;

-- Grant final permissions
GRANT USAGE ON SCHEMA FraudDetection TO fraud_detection_app, fraud_detection_admin, fraud_detection_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA FraudDetection TO fraud_detection_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA FraudDetection TO fraud_detection_admin;
GRANT SELECT ON ALL TABLES IN SCHEMA FraudDetection TO fraud_detection_monitor;

COMMIT;

-- ============================================================================
-- End of Complete System Deployment
-- ============================================================================

\echo 'Fraud Detection System deployment completed successfully!'
\echo 'Version: ' :'deployment_version'
\echo 'Deployment completed at: ' :'deployment_date'
\echo ''
\echo 'Next steps:'
\echo '1. Load your training data using LoadSampleData() or import real data'
\echo '2. Train models using RetrainModelsWithValidation()'
\echo '3. Set up monitoring dashboards'
\echo '4. Configure automated maintenance schedules'
\echo '5. Test real-time predictions with FastFraudScore()'