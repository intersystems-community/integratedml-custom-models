-- ============================================================================
-- Fraud Detection Performance Monitoring
-- ============================================================================
-- This script provides comprehensive monitoring and alerting for the fraud
-- detection system performance, model accuracy, and system health.
--
-- Author: IntegratedML Pluggable Models Team
-- Version: 1.0.0
-- ============================================================================

USE FraudDetection;

-- ============================================================================
-- 1. Model Performance Monitoring Views
-- ============================================================================

-- Real-time model performance dashboard
CREATE OR REPLACE VIEW ModelPerformanceDashboard AS
WITH recent_predictions AS (
    SELECT 
        model_version,
        fraud_probability,
        actual_fraud,
        prediction_correct,
        prediction_latency_ms,
        created_at,
        CASE 
            WHEN fraud_probability >= 0.5 THEN 1 
            ELSE 0 
        END as predicted_fraud
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
),
performance_metrics AS (
    SELECT 
        model_version,
        COUNT(*) as total_predictions,
        COUNT(*) FILTER (WHERE actual_fraud IS NOT NULL) as labeled_predictions,
        
        -- Accuracy metrics
        AVG(prediction_correct::INTEGER) as accuracy,
        
        -- Precision, Recall, F1 for fraud detection
        COUNT(*) FILTER (WHERE predicted_fraud = 1 AND actual_fraud = 1) as true_positives,
        COUNT(*) FILTER (WHERE predicted_fraud = 1 AND actual_fraud = 0) as false_positives,
        COUNT(*) FILTER (WHERE predicted_fraud = 0 AND actual_fraud = 1) as false_negatives,
        COUNT(*) FILTER (WHERE predicted_fraud = 0 AND actual_fraud = 0) as true_negatives,
        
        -- Performance metrics
        AVG(prediction_latency_ms) as avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_latency_ms) as p95_latency_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY prediction_latency_ms) as p99_latency_ms,
        MAX(prediction_latency_ms) as max_latency_ms,
        
        -- Throughput
        COUNT(*) / EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) * 3600 as predictions_per_hour
        
    FROM recent_predictions
    WHERE actual_fraud IS NOT NULL
    GROUP BY model_version
)
SELECT 
    model_version,
    total_predictions,
    labeled_predictions,
    ROUND(accuracy * 100, 2) as accuracy_percent,
    
    -- Classification metrics
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
    
    -- Precision = TP / (TP + FP)
    CASE 
        WHEN (true_positives + false_positives) > 0 
        THEN ROUND(true_positives::DECIMAL / (true_positives + false_positives) * 100, 2)
        ELSE NULL 
    END as precision_percent,
    
    -- Recall = TP / (TP + FN)  
    CASE 
        WHEN (true_positives + false_negatives) > 0
        THEN ROUND(true_positives::DECIMAL / (true_positives + false_negatives) * 100, 2)
        ELSE NULL
    END as recall_percent,
    
    -- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    CASE 
        WHEN (true_positives + false_positives) > 0 AND (true_positives + false_negatives) > 0
        THEN ROUND(2.0 * true_positives / (2.0 * true_positives + false_positives + false_negatives) * 100, 2)
        ELSE NULL
    END as f1_score_percent,
    
    -- Latency metrics
    ROUND(avg_latency_ms, 2) as avg_latency_ms,
    ROUND(p95_latency_ms, 2) as p95_latency_ms,
    ROUND(p99_latency_ms, 2) as p99_latency_ms,
    ROUND(max_latency_ms, 2) as max_latency_ms,
    
    -- Throughput
    ROUND(predictions_per_hour, 0) as predictions_per_hour,
    
    -- Status flags
    CASE WHEN avg_latency_ms <= 100 THEN 'GOOD' WHEN avg_latency_ms <= 200 THEN 'WARNING' ELSE 'CRITICAL' END as latency_status,
    CASE WHEN accuracy >= 0.95 THEN 'EXCELLENT' WHEN accuracy >= 0.9 THEN 'GOOD' WHEN accuracy >= 0.8 THEN 'FAIR' ELSE 'POOR' END as accuracy_status

FROM performance_metrics;

-- ============================================================================
-- 2. System Health Monitoring
-- ============================================================================

-- System health dashboard
CREATE OR REPLACE VIEW SystemHealthDashboard AS
WITH system_metrics AS (
    SELECT 
        -- Prediction volume trends
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour') as predictions_last_hour,
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day') as predictions_last_day,
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 week') as predictions_last_week,
        
        -- Error rates
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' AND fraud_probability IS NULL) as errors_last_hour,
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' AND fraud_probability IS NULL) as errors_last_day,
        
        -- High risk transactions
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' AND fraud_probability >= 0.8) as high_risk_last_hour,
        COUNT(*) FILTER (WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day' AND fraud_probability >= 0.8) as high_risk_last_day,
        
        -- Cache performance
        (SELECT COUNT(*) FROM PredictionCache WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour') as cache_entries_added_hour,
        (SELECT AVG(access_count) FROM PredictionCache WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day') as avg_cache_hit_rate,
        
        -- Alert counts
        (SELECT COUNT(*) FROM SystemAlerts WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour' AND resolved = FALSE) as active_alerts,
        (SELECT COUNT(*) FROM SystemAlerts WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 day') as alerts_last_day
        
    FROM PredictionLog
),
velocity_metrics AS (
    SELECT 
        COUNT(*) as active_customers,
        AVG(transactions_last_hour) as avg_customer_velocity_hour,
        MAX(transactions_last_hour) as max_customer_velocity_hour,
        COUNT(*) FILTER (WHERE transactions_last_hour > 10) as high_velocity_customers
    FROM CustomerVelocity 
    WHERE updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
)
SELECT 
    -- Volume metrics
    sm.predictions_last_hour,
    sm.predictions_last_day,
    sm.predictions_last_week,
    
    -- Quality metrics
    CASE 
        WHEN sm.predictions_last_hour > 0 
        THEN ROUND(sm.errors_last_hour::DECIMAL / sm.predictions_last_hour * 100, 2)
        ELSE 0 
    END as error_rate_percent_hour,
    
    CASE 
        WHEN sm.predictions_last_day > 0 
        THEN ROUND(sm.errors_last_day::DECIMAL / sm.predictions_last_day * 100, 2)
        ELSE 0 
    END as error_rate_percent_day,
    
    -- Risk distribution
    sm.high_risk_last_hour,
    sm.high_risk_last_day,
    CASE 
        WHEN sm.predictions_last_hour > 0
        THEN ROUND(sm.high_risk_last_hour::DECIMAL / sm.predictions_last_hour * 100, 2)
        ELSE 0
    END as high_risk_rate_percent_hour,
    
    -- Cache performance
    sm.cache_entries_added_hour,
    ROUND(COALESCE(sm.avg_cache_hit_rate, 0), 2) as avg_cache_hit_rate,
    
    -- Customer activity
    vm.active_customers,
    ROUND(vm.avg_customer_velocity_hour, 2) as avg_customer_tx_per_hour,
    vm.max_customer_velocity_hour,
    vm.high_velocity_customers,
    
    -- Alerts
    sm.active_alerts,
    sm.alerts_last_day,
    
    -- Overall system status
    CASE 
        WHEN sm.active_alerts > 5 THEN 'CRITICAL'
        WHEN sm.active_alerts > 2 OR (sm.predictions_last_hour > 0 AND sm.errors_last_hour::DECIMAL / sm.predictions_last_hour > 0.05) THEN 'WARNING'
        WHEN sm.predictions_last_hour > 0 THEN 'HEALTHY'
        ELSE 'IDLE'
    END as system_status,
    
    CURRENT_TIMESTAMP as last_updated

FROM system_metrics sm
CROSS JOIN velocity_metrics vm;

-- ============================================================================
-- 3. Fraud Detection Analytics
-- ============================================================================

-- Fraud trends and patterns view
CREATE OR REPLACE VIEW FraudAnalyticsDashboard AS
WITH fraud_trends AS (
    SELECT 
        DATE(created_at) as date,
        EXTRACT(HOUR FROM created_at) as hour,
        COUNT(*) as total_transactions,
        COUNT(*) FILTER (WHERE fraud_probability >= 0.8) as high_risk_transactions,
        COUNT(*) FILTER (WHERE fraud_probability >= 0.5) as medium_high_risk_transactions,
        COUNT(*) FILTER (WHERE actual_fraud = 1) as confirmed_fraud,
        AVG(fraud_probability) as avg_fraud_score,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fraud_probability) as p95_fraud_score
    FROM PredictionLog
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY DATE(created_at), EXTRACT(HOUR FROM created_at)
),
customer_risk_analysis AS (
    SELECT 
        customer_id,
        COUNT(*) as total_transactions,
        COUNT(*) FILTER (WHERE fraud_probability >= 0.8) as high_risk_count,
        MAX(fraud_probability) as max_fraud_score,
        AVG(fraud_probability) as avg_fraud_score,
        COUNT(*) FILTER (WHERE actual_fraud = 1) as confirmed_fraud_count
    FROM PredictionLog
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY customer_id
    HAVING COUNT(*) >= 5  -- Only customers with significant activity
),
top_risk_patterns AS (
    SELECT 
        'High Risk Customers' as pattern_type,
        COUNT(*) as pattern_count,
        AVG(max_fraud_score) as avg_max_score
    FROM customer_risk_analysis
    WHERE high_risk_count >= 3
    
    UNION ALL
    
    SELECT 
        'Confirmed Fraud Customers' as pattern_type,
        COUNT(*) as pattern_count,
        AVG(avg_fraud_score) as avg_max_score  
    FROM customer_risk_analysis
    WHERE confirmed_fraud_count >= 1
)
SELECT 
    -- Daily summary for last 7 days
    (SELECT COUNT(DISTINCT date) FROM fraud_trends) as days_analyzed,
    (SELECT SUM(total_transactions) FROM fraud_trends) as total_transactions_week,
    (SELECT SUM(high_risk_transactions) FROM fraud_trends) as high_risk_transactions_week,
    (SELECT SUM(confirmed_fraud) FROM fraud_trends) as confirmed_fraud_week,
    
    -- Risk rates
    CASE 
        WHEN (SELECT SUM(total_transactions) FROM fraud_trends) > 0
        THEN ROUND((SELECT SUM(high_risk_transactions) FROM fraud_trends)::DECIMAL / 
                   (SELECT SUM(total_transactions) FROM fraud_trends) * 100, 2)
        ELSE 0
    END as high_risk_rate_percent,
    
    CASE 
        WHEN (SELECT SUM(total_transactions) FROM fraud_trends) > 0
        THEN ROUND((SELECT SUM(confirmed_fraud) FROM fraud_trends)::DECIMAL / 
                   (SELECT SUM(total_transactions) FROM fraud_trends) * 100, 2)
        ELSE 0
    END as fraud_rate_percent,
    
    -- Peak risk periods
    (SELECT hour FROM fraud_trends WHERE high_risk_transactions = (SELECT MAX(high_risk_transactions) FROM fraud_trends) LIMIT 1) as peak_risk_hour,
    (SELECT MAX(high_risk_transactions) FROM fraud_trends) as peak_risk_hour_count,
    
    -- Customer patterns
    (SELECT COUNT(*) FROM customer_risk_analysis WHERE high_risk_count >= 3) as high_risk_customers,
    (SELECT COUNT(*) FROM customer_risk_analysis WHERE confirmed_fraud_count >= 1) as fraud_customers,
    
    -- Pattern insights
    (SELECT ROUND(AVG(avg_max_score), 4) FROM top_risk_patterns WHERE pattern_type = 'High Risk Customers') as avg_high_risk_score,
    (SELECT ROUND(AVG(avg_max_score), 4) FROM top_risk_patterns WHERE pattern_type = 'Confirmed Fraud Customers') as avg_fraud_customer_score,
    
    CURRENT_TIMESTAMP as analysis_timestamp;

-- ============================================================================
-- 4. Performance Monitoring Functions
-- ============================================================================

-- Function to generate comprehensive performance report
CREATE OR REPLACE FUNCTION GeneratePerformanceReport(
    p_hours_back INTEGER DEFAULT 24
) RETURNS TABLE (
    metric_category VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(20),
    status VARCHAR(20),
    threshold_value DECIMAL(15,4),
    details JSONB
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    
    -- Latency metrics
    WITH latency_metrics AS (
        SELECT 
            AVG(prediction_latency_ms) as avg_latency,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_latency_ms) as p95_latency,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY prediction_latency_ms) as p99_latency,
            COUNT(*) FILTER (WHERE prediction_latency_ms > 100) as high_latency_count,
            COUNT(*) as total_count
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - (p_hours_back || ' hours')::INTERVAL
        AND prediction_latency_ms IS NOT NULL
    ),
    
    -- Accuracy metrics
    accuracy_metrics AS (
        SELECT 
            AVG(prediction_correct::INTEGER) as accuracy,
            COUNT(*) FILTER (WHERE fraud_probability >= 0.5 AND actual_fraud = 1) as tp,
            COUNT(*) FILTER (WHERE fraud_probability >= 0.5 AND actual_fraud = 0) as fp,
            COUNT(*) FILTER (WHERE fraud_probability < 0.5 AND actual_fraud = 1) as fn,
            COUNT(*) FILTER (WHERE fraud_probability < 0.5 AND actual_fraud = 0) as tn
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - (p_hours_back || ' hours')::INTERVAL
        AND actual_fraud IS NOT NULL
    ),
    
    -- Throughput metrics
    throughput_metrics AS (
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(*) / p_hours_back as predictions_per_hour,
            COUNT(DISTINCT customer_id) as unique_customers
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - (p_hours_back || ' hours')::INTERVAL
    )
    
    -- Latency results
    SELECT 
        'Performance'::VARCHAR(50),
        'Average Latency'::VARCHAR(100),
        lm.avg_latency,
        'milliseconds'::VARCHAR(20),
        CASE WHEN lm.avg_latency <= 100 THEN 'GOOD' WHEN lm.avg_latency <= 200 THEN 'WARNING' ELSE 'CRITICAL' END::VARCHAR(20),
        100.0::DECIMAL(15,4),
        JSONB_BUILD_OBJECT('total_predictions', lm.total_count)
    FROM latency_metrics lm
    
    UNION ALL
    
    SELECT 
        'Performance'::VARCHAR(50),
        'P95 Latency'::VARCHAR(100),
        lm.p95_latency,
        'milliseconds'::VARCHAR(20),
        CASE WHEN lm.p95_latency <= 100 THEN 'GOOD' WHEN lm.p95_latency <= 200 THEN 'WARNING' ELSE 'CRITICAL' END::VARCHAR(20),
        100.0::DECIMAL(15,4),
        JSONB_BUILD_OBJECT('high_latency_count', lm.high_latency_count, 'total_count', lm.total_count)
    FROM latency_metrics lm
    
    UNION ALL
    
    -- Accuracy results
    SELECT 
        'Accuracy'::VARCHAR(50),
        'Overall Accuracy'::VARCHAR(100),
        am.accuracy * 100,
        'percent'::VARCHAR(20),
        CASE WHEN am.accuracy >= 0.95 THEN 'EXCELLENT' WHEN am.accuracy >= 0.9 THEN 'GOOD' WHEN am.accuracy >= 0.8 THEN 'FAIR' ELSE 'POOR' END::VARCHAR(20),
        90.0::DECIMAL(15,4),
        JSONB_BUILD_OBJECT('tp', am.tp, 'fp', am.fp, 'tn', am.tn, 'fn', am.fn)
    FROM accuracy_metrics am
    WHERE am.accuracy IS NOT NULL
    
    UNION ALL
    
    -- Precision
    SELECT 
        'Accuracy'::VARCHAR(50),
        'Precision'::VARCHAR(100),
        CASE WHEN (am.tp + am.fp) > 0 THEN am.tp::DECIMAL / (am.tp + am.fp) * 100 ELSE NULL END,
        'percent'::VARCHAR(20),
        CASE 
            WHEN (am.tp + am.fp) = 0 THEN 'N/A'
            WHEN am.tp::DECIMAL / (am.tp + am.fp) >= 0.8 THEN 'GOOD' 
            WHEN am.tp::DECIMAL / (am.tp + am.fp) >= 0.6 THEN 'FAIR' 
            ELSE 'POOR' 
        END::VARCHAR(20),
        80.0::DECIMAL(15,4),
        JSONB_BUILD_OBJECT('true_positives', am.tp, 'false_positives', am.fp)
    FROM accuracy_metrics am
    
    UNION ALL
    
    -- Throughput results
    SELECT 
        'Throughput'::VARCHAR(50),
        'Predictions Per Hour'::VARCHAR(100),
        tm.predictions_per_hour,
        'predictions/hour'::VARCHAR(20),
        CASE WHEN tm.predictions_per_hour >= 1000 THEN 'GOOD' WHEN tm.predictions_per_hour >= 100 THEN 'FAIR' ELSE 'LOW' END::VARCHAR(20),
        1000.0::DECIMAL(15,4),
        JSONB_BUILD_OBJECT('total_predictions', tm.total_predictions, 'unique_customers', tm.unique_customers)
    FROM throughput_metrics tm;
END;
$$;

-- ============================================================================
-- 5. Automated Alerting System
-- ============================================================================

-- Function to check and create performance alerts
CREATE OR REPLACE FUNCTION CheckPerformanceAlerts()
RETURNS INTEGER
LANGUAGE SQL
AS $$
DECLARE
    alert_count INTEGER := 0;
    avg_latency DECIMAL;
    p95_latency DECIMAL;
    error_rate DECIMAL;
    high_risk_rate DECIMAL;
BEGIN
    -- Check average latency
    SELECT AVG(prediction_latency_ms) 
    INTO avg_latency
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    AND prediction_latency_ms IS NOT NULL;
    
    IF avg_latency > 150 THEN
        INSERT INTO SystemAlerts (alert_type, message, severity)
        VALUES (
            'HIGH_AVERAGE_LATENCY',
            'Average prediction latency is ' || ROUND(avg_latency, 2) || 'ms (threshold: 150ms)',
            CASE WHEN avg_latency > 300 THEN 'CRITICAL' ELSE 'WARNING' END
        );
        alert_count := alert_count + 1;
    END IF;
    
    -- Check P95 latency
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_latency_ms)
    INTO p95_latency
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    AND prediction_latency_ms IS NOT NULL;
    
    IF p95_latency > 200 THEN
        INSERT INTO SystemAlerts (alert_type, message, severity)
        VALUES (
            'HIGH_P95_LATENCY',
            'P95 prediction latency is ' || ROUND(p95_latency, 2) || 'ms (threshold: 200ms)',
            CASE WHEN p95_latency > 500 THEN 'CRITICAL' ELSE 'WARNING' END
        );
        alert_count := alert_count + 1;
    END IF;
    
    -- Check error rate
    SELECT 
        COUNT(*) FILTER (WHERE fraud_probability IS NULL)::DECIMAL / COUNT(*) * 100
    INTO error_rate
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    IF error_rate > 5 THEN
        INSERT INTO SystemAlerts (alert_type, message, severity)
        VALUES (
            'HIGH_ERROR_RATE',
            'Prediction error rate is ' || ROUND(error_rate, 2) || '% (threshold: 5%)',
            CASE WHEN error_rate > 15 THEN 'CRITICAL' ELSE 'WARNING' END
        );
        alert_count := alert_count + 1;
    END IF;
    
    -- Check high risk transaction rate
    SELECT 
        COUNT(*) FILTER (WHERE fraud_probability >= 0.8)::DECIMAL / COUNT(*) * 100
    INTO high_risk_rate
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    IF high_risk_rate > 10 THEN
        INSERT INTO SystemAlerts (alert_type, message, severity)
        VALUES (
            'HIGH_RISK_SPIKE',
            'High risk transaction rate is ' || ROUND(high_risk_rate, 2) || '% (threshold: 10%)',
            'WARNING'
        );
        alert_count := alert_count + 1;
    END IF;
    
    RETURN alert_count;
END;
$$;

-- ============================================================================
-- 6. Scheduled Monitoring Jobs
-- ============================================================================

-- Procedure to run hourly performance checks
CREATE OR REPLACE PROCEDURE HourlyPerformanceCheck()
LANGUAGE SQL
AS $$
DECLARE
    alert_count INTEGER;
    performance_summary RECORD;
BEGIN
    -- Run performance alert checks
    SELECT CheckPerformanceAlerts() INTO alert_count;
    
    -- Log performance summary
    SELECT 
        COUNT(*) as predictions_count,
        AVG(prediction_latency_ms) as avg_latency,
        COUNT(*) FILTER (WHERE fraud_probability >= 0.8) as high_risk_count
    INTO performance_summary
    FROM PredictionLog 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    -- Insert performance log entry
    INSERT INTO SystemLog (
        log_type,
        message,
        details,
        created_at
    ) VALUES (
        'PERFORMANCE_CHECK',
        'Hourly performance check completed',
        JSONB_BUILD_OBJECT(
            'predictions_count', performance_summary.predictions_count,
            'avg_latency_ms', performance_summary.avg_latency,
            'high_risk_count', performance_summary.high_risk_count,
            'alerts_generated', alert_count
        ),
        CURRENT_TIMESTAMP
    );
    
    -- Clean up old cache entries (older than 1 day)
    DELETE FROM PredictionCache 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day';
    
    -- Update customer velocity data
    CALL UpdateCustomerVelocity();
    
    COMMIT;
END;
$$;

-- ============================================================================
-- 7. Supporting Tables for Monitoring
-- ============================================================================

-- System log table
CREATE TABLE IF NOT EXISTS SystemLog (
    log_id SERIAL PRIMARY KEY,
    log_type VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer profiles table (if not exists)
CREATE TABLE IF NOT EXISTS CustomerProfiles (
    customer_id VARCHAR(255) PRIMARY KEY,
    customer_tier VARCHAR(20),
    account_balance DECIMAL(15,2),
    avg_transaction_amount DECIMAL(15,2),
    transaction_frequency DECIMAL(8,4),
    customer_age_days INTEGER,
    preferred_countries TEXT,
    last_transaction_date DATE,
    total_transactions INTEGER DEFAULT 0,
    risk_score DECIMAL(8,6) DEFAULT 0.1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create monitoring indexes
CREATE INDEX IF NOT EXISTS idx_system_log_type_date ON SystemLog(log_type, created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_log_latency ON PredictionLog(prediction_latency_ms) WHERE prediction_latency_ms IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prediction_log_fraud_prob ON PredictionLog(fraud_probability) WHERE fraud_probability IS NOT NULL;

-- ============================================================================
-- 8. Monitoring Dashboard Refresh
-- ============================================================================

-- Procedure to refresh all monitoring views
CREATE OR REPLACE PROCEDURE RefreshMonitoringDashboards()
LANGUAGE SQL
AS $$
BEGIN
    -- Refresh materialized views if any exist
    REFRESH MATERIALIZED VIEW CONCURRENTLY FastCustomerLookup;
    
    -- Log refresh
    INSERT INTO SystemLog (log_type, message, created_at)
    VALUES ('DASHBOARD_REFRESH', 'Monitoring dashboards refreshed', CURRENT_TIMESTAMP);
    
    COMMIT;
END;
$$;

-- Grant permissions for monitoring users
GRANT SELECT ON ModelPerformanceDashboard TO fraud_detection_monitor;
GRANT SELECT ON SystemHealthDashboard TO fraud_detection_monitor;
GRANT SELECT ON FraudAnalyticsDashboard TO fraud_detection_monitor;
GRANT EXECUTE ON FUNCTION GeneratePerformanceReport TO fraud_detection_monitor;
GRANT SELECT ON SystemAlerts TO fraud_detection_monitor;
GRANT SELECT ON SystemLog TO fraud_detection_monitor;

-- ============================================================================
-- End of Performance Monitoring Script
-- ============================================================================