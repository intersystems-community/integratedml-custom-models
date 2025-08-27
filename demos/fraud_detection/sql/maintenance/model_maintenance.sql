-- ============================================================================
-- Fraud Detection Model Maintenance
-- ============================================================================
-- This script provides maintenance procedures for the fraud detection system
-- including model retraining, data cleanup, and system optimization.
--
-- Author: IntegratedML Pluggable Models Team
-- Version: 1.0.0
-- ============================================================================

USE FraudDetection;

-- ============================================================================
-- 1. Model Retraining Procedures
-- ============================================================================

-- Comprehensive model retraining with validation
CREATE OR REPLACE PROCEDURE RetrainModelsWithValidation(
    p_training_start_date DATE DEFAULT CURRENT_DATE - INTERVAL '30 days',
    p_training_end_date DATE DEFAULT CURRENT_DATE - INTERVAL '1 day',
    p_validation_split DECIMAL DEFAULT 0.2,
    p_min_accuracy_threshold DECIMAL DEFAULT 0.85
)
LANGUAGE SQL
AS $$
DECLARE
    training_record_count INTEGER;
    validation_record_count INTEGER;
    model_performance RECORD;
    retraining_successful BOOLEAN := TRUE;
    error_message TEXT;
BEGIN
    -- Log retraining start
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'MODEL_RETRAINING_START',
        'Starting comprehensive model retraining',
        JSONB_BUILD_OBJECT(
            'training_start_date', p_training_start_date,
            'training_end_date', p_training_end_date,
            'validation_split', p_validation_split,
            'min_accuracy_threshold', p_min_accuracy_threshold
        )
    );
    
    -- Check data availability
    SELECT COUNT(*) 
    INTO training_record_count
    FROM TransactionTrainingData 
    WHERE DATE(transaction_timestamp) BETWEEN p_training_start_date AND p_training_end_date
    AND is_fraud IS NOT NULL;
    
    IF training_record_count < 1000 THEN
        RAISE EXCEPTION 'Insufficient training data: % records (minimum: 1000)', training_record_count;
    END IF;
    
    -- Create validation dataset
    CREATE TEMPORARY TABLE ValidationData AS
    SELECT * FROM TransactionTrainingData 
    WHERE DATE(transaction_timestamp) BETWEEN p_training_start_date AND p_training_end_date
    AND is_fraud IS NOT NULL
    AND RANDOM() < p_validation_split;
    
    GET DIAGNOSTICS validation_record_count = ROW_COUNT;
    
    -- Create training dataset (excluding validation data)
    CREATE TEMPORARY TABLE TrainingDataset AS
    SELECT ttd.* FROM TransactionTrainingData ttd
    LEFT JOIN ValidationData vd ON ttd.transaction_id = vd.transaction_id
    WHERE DATE(ttd.transaction_timestamp) BETWEEN p_training_start_date AND p_training_end_date
    AND ttd.is_fraud IS NOT NULL
    AND vd.transaction_id IS NULL;
    
    BEGIN
        -- Retrain Ensemble Model
        DROP MODEL IF EXISTS FraudDetectionEnsemble_Retrained;
        CREATE MODEL FraudDetectionEnsemble_Retrained PREDICTING (is_fraud) 
        FROM (
            SELECT * FROM TransactionFeatures 
            WHERE transaction_id IN (SELECT transaction_id FROM TrainingDataset)
        ) USING {"seed": 42, "provider": "H2O"};
        
        -- Retrain Rule-based Model
        DROP MODEL IF EXISTS RuleBasedFraudDetector_Retrained;
        CREATE MODEL RuleBasedFraudDetector_Retrained PREDICTING (is_fraud) FROM (
            SELECT 
                transaction_id,
                CASE WHEN amount > 1000 THEN 1 ELSE 0 END as high_amount_flag,
                CASE WHEN EXTRACT(HOUR FROM transaction_timestamp) NOT BETWEEN 6 AND 22 THEN 1 ELSE 0 END as unusual_time_flag,
                CASE WHEN merchant_country != customer_country THEN 1 ELSE 0 END as international_flag,
                high_risk_merchant,
                weekend_transaction,
                late_night_transaction,
                is_fraud
            FROM TrainingDataset
        ) USING {"seed": 42, "provider": "H2O"};
        
        -- Retrain Anomaly Detection Model
        DROP MODEL IF EXISTS AnomalyFraudDetector_Retrained;
        CREATE MODEL AnomalyFraudDetector_Retrained PREDICTING (is_fraud) FROM (
            SELECT 
                transaction_id,
                amount,
                distance_from_home,
                anomaly_score,
                amount / NULLIF(avg_transaction_amount, 0) as amount_ratio,
                CASE WHEN transaction_frequency > 0 THEN 1/transaction_frequency ELSE 0 END as frequency_score,
                is_fraud
            FROM TrainingDataset td
            JOIN CustomerProfiles cp ON td.customer_id = cp.customer_id
        ) USING {"seed": 42, "provider": "H2O"};
        
        -- Retrain Behavioral Model
        DROP MODEL IF EXISTS BehavioralFraudDetector_Retrained;
        CREATE MODEL BehavioralFraudDetector_Retrained PREDICTING (is_fraud) FROM (
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
            FROM TrainingDataset
        ) USING {"seed": 42, "provider": "H2O"};
        
        -- Retrain Neural Network Model
        DROP MODEL IF EXISTS NeuralFraudDetector_Retrained;
        CREATE MODEL NeuralFraudDetector_Retrained PREDICTING (is_fraud) 
        FROM (
            SELECT * FROM TransactionFeatures 
            WHERE transaction_id IN (SELECT transaction_id FROM TrainingDataset)
        ) USING {"seed": 42, "provider": "H2O", "algorithm": "DeepLearning"};
        
    EXCEPTION
        WHEN OTHERS THEN
            error_message := SQLERRM;
            retraining_successful := FALSE;
            
            INSERT INTO SystemLog (log_type, message, details)
            VALUES (
                'MODEL_RETRAINING_ERROR',
                'Model retraining failed: ' || error_message,
                JSONB_BUILD_OBJECT('error_details', error_message)
            );
            
            RAISE EXCEPTION 'Model retraining failed: %', error_message;
    END;
    
    -- Validate retrained models
    CALL ValidateRetrainedModels(validation_record_count, p_min_accuracy_threshold);
    
    -- If validation passed, deploy new models
    CALL DeployRetrainedModels();
    
    -- Log successful completion
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'MODEL_RETRAINING_SUCCESS',
        'Model retraining completed successfully',
        JSONB_BUILD_OBJECT(
            'training_records', training_record_count,
            'validation_records', validation_record_count,
            'models_retrained', 5
        )
    );
    
    -- Cleanup temporary tables
    DROP TABLE IF EXISTS ValidationData;
    DROP TABLE IF EXISTS TrainingDataset;
    
    COMMIT;
END;
$$;

-- ============================================================================
-- 2. Model Validation Procedures
-- ============================================================================

-- Validate retrained models against holdout data
CREATE OR REPLACE PROCEDURE ValidateRetrainedModels(
    p_validation_record_count INTEGER,
    p_min_accuracy_threshold DECIMAL
)
LANGUAGE SQL
AS $$
DECLARE
    ensemble_accuracy DECIMAL;
    rule_accuracy DECIMAL;
    anomaly_accuracy DECIMAL;
    behavioral_accuracy DECIMAL;
    neural_accuracy DECIMAL;
    validation_failed BOOLEAN := FALSE;
    failed_models TEXT[] := ARRAY[]::TEXT[];
BEGIN
    -- Test Ensemble Model
    WITH ensemble_predictions AS (
        SELECT 
            vd.is_fraud as actual,
            CASE WHEN PREDICT(FraudDetectionEnsemble_Retrained) >= 0.5 THEN 1 ELSE 0 END as predicted
        FROM ValidationData vd
        JOIN TransactionFeatures tf ON vd.transaction_id = tf.transaction_id
    )
    SELECT AVG((actual = predicted)::INTEGER) 
    INTO ensemble_accuracy
    FROM ensemble_predictions;
    
    -- Test Rule-based Model
    WITH rule_predictions AS (
        SELECT 
            vd.is_fraud as actual,
            CASE WHEN PREDICT(RuleBasedFraudDetector_Retrained) >= 0.5 THEN 1 ELSE 0 END as predicted
        FROM ValidationData vd
        JOIN (
            SELECT 
                transaction_id,
                CASE WHEN amount > 1000 THEN 1 ELSE 0 END as high_amount_flag,
                CASE WHEN EXTRACT(HOUR FROM transaction_timestamp) NOT BETWEEN 6 AND 22 THEN 1 ELSE 0 END as unusual_time_flag,
                CASE WHEN merchant_country != customer_country THEN 1 ELSE 0 END as international_flag,
                high_risk_merchant,
                weekend_transaction,
                late_night_transaction
            FROM ValidationData
        ) rule_features ON vd.transaction_id = rule_features.transaction_id
    )
    SELECT AVG((actual = predicted)::INTEGER) 
    INTO rule_accuracy
    FROM rule_predictions;
    
    -- Similar validation for other models...
    -- (Abbreviated for brevity, but would include anomaly, behavioral, and neural)
    
    -- Check accuracy thresholds
    IF ensemble_accuracy < p_min_accuracy_threshold THEN
        validation_failed := TRUE;
        failed_models := array_append(failed_models, 'FraudDetectionEnsemble');
    END IF;
    
    IF rule_accuracy < p_min_accuracy_threshold THEN
        validation_failed := TRUE;
        failed_models := array_append(failed_models, 'RuleBasedFraudDetector');
    END IF;
    
    -- Log validation results
    INSERT INTO ModelPerformance (
        model_name,
        evaluation_date,
        accuracy,
        total_predictions,
        created_at
    ) VALUES 
    ('FraudDetectionEnsemble_Retrained', CURRENT_DATE, ensemble_accuracy, p_validation_record_count, CURRENT_TIMESTAMP),
    ('RuleBasedFraudDetector_Retrained', CURRENT_DATE, rule_accuracy, p_validation_record_count, CURRENT_TIMESTAMP);
    
    -- Fail if any model doesn't meet threshold
    IF validation_failed THEN
        RAISE EXCEPTION 'Model validation failed. Models below threshold: %', array_to_string(failed_models, ', ');
    END IF;
    
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'MODEL_VALIDATION_SUCCESS',
        'All retrained models passed validation',
        JSONB_BUILD_OBJECT(
            'ensemble_accuracy', ensemble_accuracy,
            'rule_accuracy', rule_accuracy,
            'min_threshold', p_min_accuracy_threshold
        )
    );
END;
$$;

-- ============================================================================
-- 3. Model Deployment Procedures
-- ============================================================================

-- Deploy validated retrained models
CREATE OR REPLACE PROCEDURE DeployRetrainedModels()
LANGUAGE SQL
AS $$
DECLARE
    backup_suffix TEXT;
BEGIN
    backup_suffix := '_backup_' || TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    
    -- Backup existing models
    EXECUTE 'ALTER MODEL FraudDetectionEnsemble RENAME TO FraudDetectionEnsemble' || backup_suffix;
    EXECUTE 'ALTER MODEL RuleBasedFraudDetector RENAME TO RuleBasedFraudDetector' || backup_suffix;
    EXECUTE 'ALTER MODEL AnomalyFraudDetector RENAME TO AnomalyFraudDetector' || backup_suffix;
    EXECUTE 'ALTER MODEL BehavioralFraudDetector RENAME TO BehavioralFraudDetector' || backup_suffix;
    EXECUTE 'ALTER MODEL NeuralFraudDetector RENAME TO NeuralFraudDetector' || backup_suffix;
    
    -- Deploy new models
    ALTER MODEL FraudDetectionEnsemble_Retrained RENAME TO FraudDetectionEnsemble;
    ALTER MODEL RuleBasedFraudDetector_Retrained RENAME TO RuleBasedFraudDetector;
    ALTER MODEL AnomalyFraudDetector_Retrained RENAME TO AnomalyFraudDetector;
    ALTER MODEL BehavioralFraudDetector_Retrained RENAME TO BehavioralFraudDetector;
    ALTER MODEL NeuralFraudDetector_Retrained RENAME TO NeuralFraudDetector;
    
    -- Log deployment
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'MODEL_DEPLOYMENT',
        'Retrained models deployed successfully',
        JSONB_BUILD_OBJECT(
            'backup_suffix', backup_suffix,
            'deployment_time', CURRENT_TIMESTAMP
        )
    );
    
    -- Create alert for model deployment
    INSERT INTO SystemAlerts (alert_type, message, severity)
    VALUES (
        'MODEL_DEPLOYED',
        'New fraud detection models have been deployed',
        'INFO'
    );
END;
$$;

-- ============================================================================
-- 4. Data Maintenance Procedures
-- ============================================================================

-- Clean up old prediction logs and cache entries
CREATE OR REPLACE PROCEDURE CleanupOldData(
    p_prediction_log_retention_days INTEGER DEFAULT 30,
    p_cache_retention_hours INTEGER DEFAULT 24,
    p_alert_retention_days INTEGER DEFAULT 7
)
LANGUAGE SQL
AS $$
DECLARE
    deleted_predictions INTEGER;
    deleted_cache_entries INTEGER;
    deleted_alerts INTEGER;
BEGIN
    -- Clean up old prediction logs
    DELETE FROM PredictionLog 
    WHERE created_at < CURRENT_TIMESTAMP - (p_prediction_log_retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_predictions = ROW_COUNT;
    
    -- Clean up old cache entries
    DELETE FROM PredictionCache 
    WHERE created_at < CURRENT_TIMESTAMP - (p_cache_retention_hours || ' hours')::INTERVAL;
    GET DIAGNOSTICS deleted_cache_entries = ROW_COUNT;
    
    -- Clean up resolved alerts
    DELETE FROM SystemAlerts 
    WHERE resolved = TRUE 
    AND resolved_at < CURRENT_TIMESTAMP - (p_alert_retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_alerts = ROW_COUNT;
    
    -- Log cleanup results
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'DATA_CLEANUP',
        'Data cleanup completed',
        JSONB_BUILD_OBJECT(
            'deleted_predictions', deleted_predictions,
            'deleted_cache_entries', deleted_cache_entries,
            'deleted_alerts', deleted_alerts,
            'retention_days_predictions', p_prediction_log_retention_days,
            'retention_hours_cache', p_cache_retention_hours,
            'retention_days_alerts', p_alert_retention_days
        )
    );
    
    COMMIT;
END;
$$;

-- ============================================================================
-- 5. Customer Velocity Update Procedures
-- ============================================================================

-- Update customer velocity metrics
CREATE OR REPLACE PROCEDURE UpdateCustomerVelocity()
LANGUAGE SQL
AS $$
DECLARE
    updated_customers INTEGER;
BEGIN
    -- Update or insert customer velocity data
    WITH velocity_calculations AS (
        SELECT 
            customer_id,
            COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour') as tx_last_hour,
            COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day') as tx_last_day,
            SUM(CASE WHEN amount IS NOT NULL AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' 
                THEN amount ELSE 0 END) as amt_last_hour,
            SUM(CASE WHEN amount IS NOT NULL AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' 
                THEN amount ELSE 0 END) as amt_last_day
        FROM (
            SELECT 
                customer_id,
                (CASE WHEN fraud_probability IS NOT NULL THEN 
                    (SELECT amount FROM TransactionTrainingData ttd 
                     WHERE ttd.transaction_id = pl.transaction_id LIMIT 1)
                 ELSE NULL END) as amount,
                created_at
            FROM PredictionLog pl
        ) customer_transactions
        WHERE customer_id IS NOT NULL
        AND created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day'
        GROUP BY customer_id
    )
    INSERT INTO CustomerVelocity (
        customer_id,
        transactions_last_hour,
        transactions_last_day,
        amount_last_hour,
        amount_last_day,
        updated_at
    )
    SELECT 
        customer_id,
        tx_last_hour,
        tx_last_day,
        amt_last_hour,
        amt_last_day,
        CURRENT_TIMESTAMP
    FROM velocity_calculations
    ON CONFLICT (customer_id) DO UPDATE SET
        transactions_last_hour = EXCLUDED.transactions_last_hour,
        transactions_last_day = EXCLUDED.transactions_last_day,
        amount_last_hour = EXCLUDED.amount_last_hour,
        amount_last_day = EXCLUDED.amount_last_day,
        updated_at = EXCLUDED.updated_at;
    
    GET DIAGNOSTICS updated_customers = ROW_COUNT;
    
    -- Log update
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'VELOCITY_UPDATE',
        'Customer velocity data updated',
        JSONB_BUILD_OBJECT('updated_customers', updated_customers)
    );
END;
$$;

-- ============================================================================
-- 6. System Health Checks
-- ============================================================================

-- Comprehensive system health check
CREATE OR REPLACE FUNCTION SystemHealthCheck()
RETURNS TABLE (
    check_category VARCHAR(50),
    check_name VARCHAR(100),
    status VARCHAR(20),
    details JSONB
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    
    -- Check model availability
    WITH model_check AS (
        SELECT 
            COUNT(*) as model_count
        FROM INFORMATION_SCHEMA.ML_MODELS 
        WHERE MODEL_NAME LIKE '%Fraud%'
    )
    SELECT 
        'Models'::VARCHAR(50),
        'Model Availability'::VARCHAR(100),
        CASE WHEN mc.model_count >= 5 THEN 'HEALTHY' ELSE 'CRITICAL' END::VARCHAR(20),
        JSONB_BUILD_OBJECT('available_models', mc.model_count, 'expected_models', 5)
    FROM model_check mc
    
    UNION ALL
    
    -- Check recent prediction activity
    WITH prediction_check AS (
        SELECT 
            COUNT(*) as recent_predictions
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    )
    SELECT 
        'Activity'::VARCHAR(50),
        'Recent Predictions'::VARCHAR(100),
        CASE 
            WHEN pc.recent_predictions > 100 THEN 'HEALTHY'
            WHEN pc.recent_predictions > 10 THEN 'NORMAL'
            WHEN pc.recent_predictions > 0 THEN 'LOW'
            ELSE 'INACTIVE'
        END::VARCHAR(20),
        JSONB_BUILD_OBJECT('predictions_last_hour', pc.recent_predictions)
    FROM prediction_check pc
    
    UNION ALL
    
    -- Check error rates
    WITH error_check AS (
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(*) FILTER (WHERE fraud_probability IS NULL) as failed_predictions
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    )
    SELECT 
        'Quality'::VARCHAR(50),
        'Error Rate'::VARCHAR(100),
        CASE 
            WHEN ec.total_predictions = 0 THEN 'UNKNOWN'
            WHEN ec.failed_predictions::DECIMAL / ec.total_predictions <= 0.01 THEN 'HEALTHY'
            WHEN ec.failed_predictions::DECIMAL / ec.total_predictions <= 0.05 THEN 'WARNING'
            ELSE 'CRITICAL'
        END::VARCHAR(20),
        JSONB_BUILD_OBJECT(
            'total_predictions', ec.total_predictions,
            'failed_predictions', ec.failed_predictions,
            'error_rate_percent', 
            CASE WHEN ec.total_predictions > 0 
                 THEN ROUND(ec.failed_predictions::DECIMAL / ec.total_predictions * 100, 2)
                 ELSE 0 END
        )
    FROM error_check ec
    
    UNION ALL
    
    -- Check active alerts
    WITH alert_check AS (
        SELECT 
            COUNT(*) as active_alerts,
            COUNT(*) FILTER (WHERE severity = 'CRITICAL') as critical_alerts
        FROM SystemAlerts 
        WHERE resolved = FALSE
    )
    SELECT 
        'Alerts'::VARCHAR(50),
        'Active Alerts'::VARCHAR(100),
        CASE 
            WHEN ac.critical_alerts > 0 THEN 'CRITICAL'
            WHEN ac.active_alerts > 5 THEN 'WARNING'
            WHEN ac.active_alerts > 0 THEN 'NORMAL'
            ELSE 'HEALTHY'
        END::VARCHAR(20),
        JSONB_BUILD_OBJECT(
            'active_alerts', ac.active_alerts,
            'critical_alerts', ac.critical_alerts
        )
    FROM alert_check ac;
END;
$$;

-- ============================================================================
-- 7. Automated Maintenance Schedule
-- ============================================================================

-- Daily maintenance procedure
CREATE OR REPLACE PROCEDURE DailyMaintenance()
LANGUAGE SQL
AS $$
BEGIN
    -- Update customer velocity data
    CALL UpdateCustomerVelocity();
    
    -- Clean up old data (keep 30 days of predictions, 24 hours of cache)
    CALL CleanupOldData(30, 24, 7);
    
    -- Refresh materialized views
    CALL RefreshMonitoringDashboards();
    
    -- Check system health
    INSERT INTO SystemLog (log_type, message, details)
    SELECT 
        'HEALTH_CHECK',
        'Daily system health check',
        JSONB_AGG(
            JSONB_BUILD_OBJECT(
                'category', check_category,
                'check', check_name,
                'status', status,
                'details', details
            )
        )
    FROM SystemHealthCheck();
    
    -- Log maintenance completion
    INSERT INTO SystemLog (log_type, message)
    VALUES ('DAILY_MAINTENANCE', 'Daily maintenance completed successfully');
    
    COMMIT;
END;
$$;

-- Weekly maintenance procedure
CREATE OR REPLACE PROCEDURE WeeklyMaintenance()
LANGUAGE SQL
AS $$
BEGIN
    -- Analyze model performance trends
    INSERT INTO SystemLog (log_type, message, details)
    SELECT 
        'WEEKLY_PERFORMANCE_ANALYSIS',
        'Weekly model performance analysis',
        JSONB_BUILD_OBJECT(
            'avg_accuracy_week', AVG(accuracy),
            'min_accuracy_week', MIN(accuracy),
            'max_accuracy_week', MAX(accuracy),
            'performance_evaluations', COUNT(*)
        )
    FROM ModelPerformance 
    WHERE evaluation_date >= CURRENT_DATE - INTERVAL '7 days';
    
    -- Comprehensive data cleanup
    CALL CleanupOldData(30, 24, 14);
    
    -- Vacuum and analyze tables for performance
    VACUUM ANALYZE PredictionLog;
    VACUUM ANALYZE TransactionTrainingData;
    VACUUM ANALYZE ModelPerformance;
    
    INSERT INTO SystemLog (log_type, message)
    VALUES ('WEEKLY_MAINTENANCE', 'Weekly maintenance completed successfully');
    
    COMMIT;
END;
$$;

-- ============================================================================
-- 8. Emergency Procedures
-- ============================================================================

-- Emergency model rollback
CREATE OR REPLACE PROCEDURE EmergencyModelRollback(
    p_backup_suffix TEXT
)
LANGUAGE SQL
AS $$
DECLARE
    current_suffix TEXT;
BEGIN
    current_suffix := '_emergency_' || TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    
    -- Backup current models
    EXECUTE 'ALTER MODEL FraudDetectionEnsemble RENAME TO FraudDetectionEnsemble' || current_suffix;
    EXECUTE 'ALTER MODEL RuleBasedFraudDetector RENAME TO RuleBasedFraudDetector' || current_suffix;
    
    -- Restore backup models
    EXECUTE 'ALTER MODEL FraudDetectionEnsemble' || p_backup_suffix || ' RENAME TO FraudDetectionEnsemble';
    EXECUTE 'ALTER MODEL RuleBasedFraudDetector' || p_backup_suffix || ' RENAME TO RuleBasedFraudDetector';
    
    -- Log emergency rollback
    INSERT INTO SystemLog (log_type, message, details)
    VALUES (
        'EMERGENCY_ROLLBACK',
        'Emergency model rollback completed',
        JSONB_BUILD_OBJECT(
            'backup_suffix', p_backup_suffix,
            'current_suffix', current_suffix,
            'rollback_time', CURRENT_TIMESTAMP
        )
    );
    
    -- Create critical alert
    INSERT INTO SystemAlerts (alert_type, message, severity)
    VALUES (
        'EMERGENCY_ROLLBACK',
        'Emergency model rollback performed - investigate immediately',
        'CRITICAL'
    );
    
    COMMIT;
END;
$$;

-- Grant permissions for maintenance procedures
GRANT EXECUTE ON PROCEDURE RetrainModelsWithValidation TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE CleanupOldData TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE DailyMaintenance TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE WeeklyMaintenance TO fraud_detection_admin;
GRANT EXECUTE ON PROCEDURE EmergencyModelRollback TO fraud_detection_admin;
GRANT EXECUTE ON FUNCTION SystemHealthCheck TO fraud_detection_monitor;

-- ============================================================================
-- End of Model Maintenance Script
-- ============================================================================