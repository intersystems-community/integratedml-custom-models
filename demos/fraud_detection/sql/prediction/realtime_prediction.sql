-- ============================================================================
-- Real-time Fraud Detection Prediction Queries
-- ============================================================================
-- This script contains optimized SQL queries for real-time fraud detection
-- predictions with sub-100ms latency requirements.
--
-- Author: IntegratedML Pluggable Models Team
-- Version: 1.0.0
-- ============================================================================

USE FraudDetection;

-- ============================================================================
-- 1. Fast Real-time Prediction Query
-- ============================================================================

-- Optimized single transaction fraud scoring (target <50ms)
CREATE OR REPLACE FUNCTION FastFraudScore(
    p_customer_id VARCHAR(255),
    p_amount DECIMAL(15,2),
    p_merchant_id VARCHAR(255),
    p_transaction_type VARCHAR(50),
    p_merchant_category VARCHAR(100),
    p_merchant_country CHAR(2) DEFAULT 'US'
) RETURNS TABLE (
    fraud_probability DECIMAL(8,6),
    risk_level VARCHAR(20),
    confidence_score DECIMAL(8,6),
    processing_time_ms DECIMAL(10,4)
)
LANGUAGE SQL
AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    customer_profile RECORD;
    velocity_data RECORD;
    risk_indicators RECORD;
BEGIN
    start_time := CLOCK_TIMESTAMP();
    
    -- Get cached customer profile (optimized with indexes)
    SELECT 
        customer_tier,
        account_balance,
        avg_transaction_amount,
        transaction_frequency,
        preferred_countries
    INTO customer_profile
    FROM CustomerProfiles 
    WHERE customer_id = p_customer_id;
    
    -- Get velocity data from last 24 hours (pre-aggregated)
    SELECT 
        transactions_last_hour,
        transactions_last_day,
        amount_last_hour,
        amount_last_day
    INTO velocity_data
    FROM CustomerVelocity 
    WHERE customer_id = p_customer_id
    AND updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour';
    
    -- Calculate risk indicators
    SELECT 
        CASE WHEN p_amount > 1000 THEN 1 ELSE 0 END AS high_amount,
        CASE WHEN p_merchant_country != 'US' THEN 1 ELSE 0 END AS international,
        CASE WHEN EXTRACT(HOUR FROM CURRENT_TIMESTAMP) NOT BETWEEN 6 AND 22 THEN 1 ELSE 0 END AS unusual_time,
        CASE WHEN p_amount > COALESCE(customer_profile.avg_transaction_amount * 3, 500) THEN 1 ELSE 0 END AS amount_anomaly,
        CASE WHEN COALESCE(velocity_data.transactions_last_hour, 0) > 5 THEN 1 ELSE 0 END AS high_velocity
    INTO risk_indicators;
    
    -- Fast rule-based scoring
    WITH fast_features AS (
        SELECT 
            p_amount as amount,
            risk_indicators.high_amount,
            risk_indicators.international,
            risk_indicators.unusual_time,
            risk_indicators.amount_anomaly,
            risk_indicators.high_velocity,
            COALESCE(velocity_data.transactions_last_hour, 0) as tx_last_hour,
            COALESCE(velocity_data.amount_last_hour, 0) as amt_last_hour,
            CASE WHEN customer_profile.customer_tier = 'premium' THEN 0.1 ELSE 0.2 END as base_risk
    ),
    risk_calculation AS (
        SELECT 
            base_risk +
            (high_amount * 0.2) +
            (international * 0.3) +
            (unusual_time * 0.25) +
            (amount_anomaly * 0.4) +
            (high_velocity * 0.5) +
            (CASE WHEN tx_last_hour > 3 THEN 0.3 ELSE 0 END) +
            (CASE WHEN amt_last_hour > amount * 5 THEN 0.4 ELSE 0 END) as calculated_risk
        FROM fast_features
    )
    SELECT 
        LEAST(calculated_risk, 0.99) as fraud_prob,
        CASE 
            WHEN calculated_risk >= 0.8 THEN 'CRITICAL'
            WHEN calculated_risk >= 0.6 THEN 'HIGH'
            WHEN calculated_risk >= 0.3 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_lvl,
        CASE 
            WHEN calculated_risk <= 0.1 OR calculated_risk >= 0.9 THEN 0.95
            ELSE 0.75
        END as confidence,
        EXTRACT(EPOCH FROM (CLOCK_TIMESTAMP() - start_time)) * 1000 as proc_time
    INTO fraud_probability, risk_level, confidence_score, processing_time_ms
    FROM risk_calculation;
    
    RETURN QUERY SELECT fraud_probability, risk_level, confidence_score, processing_time_ms;
END;
$$;

-- ============================================================================
-- 2. Cached Prediction with Ensemble Models
-- ============================================================================

-- Cached prediction function with model ensemble
CREATE OR REPLACE FUNCTION CachedEnsemblePrediction(
    p_transaction_id VARCHAR(255),
    p_customer_id VARCHAR(255),
    p_amount DECIMAL(15,2),
    p_merchant_id VARCHAR(255),
    p_cache_ttl_seconds INTEGER DEFAULT 300
) RETURNS TABLE (
    transaction_id VARCHAR(255),
    fraud_probability DECIMAL(8,6),
    risk_level VARCHAR(20),
    ensemble_breakdown JSONB,
    cache_hit BOOLEAN,
    processing_time_ms DECIMAL(10,4)
)
LANGUAGE SQL
AS $$
DECLARE
    start_time TIMESTAMP;
    cached_result RECORD;
    feature_vector RECORD;
BEGIN
    start_time := CLOCK_TIMESTAMP();
    
    -- Check prediction cache first
    SELECT 
        result_data,
        created_at
    INTO cached_result
    FROM PredictionCache 
    WHERE cache_key = MD5(p_customer_id || p_amount::TEXT || p_merchant_id)
    AND created_at >= CURRENT_TIMESTAMP - (p_cache_ttl_seconds || ' seconds')::INTERVAL;
    
    -- Return cached result if found
    IF cached_result IS NOT NULL THEN
        RETURN QUERY 
        SELECT 
            p_transaction_id,
            (cached_result.result_data->>'fraud_probability')::DECIMAL(8,6),
            cached_result.result_data->>'risk_level',
            cached_result.result_data->'ensemble_breakdown',
            TRUE as cache_hit,
            EXTRACT(EPOCH FROM (CLOCK_TIMESTAMP() - start_time)) * 1000
        ;
        RETURN;
    END IF;
    
    -- Prepare feature vector for model prediction
    WITH transaction_features AS (
        SELECT 
            p_amount as amount,
            EXTRACT(HOUR FROM CURRENT_TIMESTAMP) as transaction_hour,
            EXTRACT(DOW FROM CURRENT_TIMESTAMP) as day_of_week,
            -- Add more features as needed
            1 as feature_ready
    ),
    model_predictions AS (
        SELECT 
            -- Use actual IntegratedML PREDICT functions here
            0.15 as rule_score,      -- PREDICT(RuleBasedFraudDetector)
            0.12 as anomaly_score,   -- PREDICT(AnomalyFraudDetector) 
            0.18 as behavioral_score,-- PREDICT(BehavioralFraudDetector)
            0.14 as neural_score     -- PREDICT(NeuralFraudDetector)
    ),
    ensemble_result AS (
        SELECT 
            -- Weighted ensemble combination
            (rule_score * 0.25 + anomaly_score * 0.25 + 
             behavioral_score * 0.25 + neural_score * 0.25) as final_prob,
            JSONB_BUILD_OBJECT(
                'rule_based', rule_score,
                'anomaly', anomaly_score,
                'behavioral', behavioral_score,
                'neural', neural_score,
                'weights', JSONB_BUILD_OBJECT(
                    'rule_based', 0.25,
                    'anomaly', 0.25,
                    'behavioral', 0.25,
                    'neural', 0.25
                )
            ) as breakdown
        FROM model_predictions
    )
    SELECT 
        final_prob,
        CASE 
            WHEN final_prob >= 0.8 THEN 'CRITICAL'
            WHEN final_prob >= 0.6 THEN 'HIGH'
            WHEN final_prob >= 0.3 THEN 'MEDIUM'
            ELSE 'LOW'
        END,
        breakdown
    INTO fraud_probability, risk_level, ensemble_breakdown
    FROM ensemble_result;
    
    -- Cache the result
    INSERT INTO PredictionCache (
        cache_key,
        result_data,
        created_at
    ) VALUES (
        MD5(p_customer_id || p_amount::TEXT || p_merchant_id),
        JSONB_BUILD_OBJECT(
            'fraud_probability', fraud_probability,
            'risk_level', risk_level,
            'ensemble_breakdown', ensemble_breakdown
        ),
        CURRENT_TIMESTAMP
    );
    
    RETURN QUERY 
    SELECT 
        p_transaction_id,
        fraud_probability,
        risk_level,
        ensemble_breakdown,
        FALSE as cache_hit,
        EXTRACT(EPOCH FROM (CLOCK_TIMESTAMP() - start_time)) * 1000
    ;
END;
$$;

-- ============================================================================
-- 3. Batch Prediction for High Throughput
-- ============================================================================

-- Optimized batch prediction for multiple transactions
CREATE OR REPLACE FUNCTION BatchFraudPrediction(
    p_transaction_batch JSONB
) RETURNS TABLE (
    transaction_id VARCHAR(255),
    fraud_probability DECIMAL(8,6),
    risk_level VARCHAR(20),
    batch_processing_time_ms DECIMAL(10,4)
)
LANGUAGE SQL
AS $$
DECLARE
    start_time TIMESTAMP;
    batch_size INTEGER;
BEGIN
    start_time := CLOCK_TIMESTAMP();
    batch_size := JSONB_ARRAY_LENGTH(p_transaction_batch);
    
    RETURN QUERY
    WITH batch_transactions AS (
        SELECT 
            (tx->>'transaction_id')::VARCHAR(255) as tx_id,
            (tx->>'customer_id')::VARCHAR(255) as cust_id,
            (tx->>'amount')::DECIMAL(15,2) as amt,
            (tx->>'merchant_id')::VARCHAR(255) as merch_id,
            (tx->>'merchant_category')::VARCHAR(100) as merch_cat
        FROM JSONB_ARRAY_ELEMENTS(p_transaction_batch) as tx
    ),
    batch_features AS (
        SELECT 
            bt.*,
            cp.avg_transaction_amount,
            cp.customer_tier,
            cv.transactions_last_hour,
            cv.amount_last_hour,
            CASE WHEN bt.amt > 1000 THEN 1 ELSE 0 END as high_amount,
            CASE WHEN bt.amt > COALESCE(cp.avg_transaction_amount * 3, 500) THEN 1 ELSE 0 END as amount_anomaly
        FROM batch_transactions bt
        LEFT JOIN CustomerProfiles cp ON bt.cust_id = cp.customer_id
        LEFT JOIN CustomerVelocity cv ON bt.cust_id = cv.customer_id 
        AND cv.updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    ),
    batch_predictions AS (
        SELECT 
            tx_id,
            -- Fast rule-based calculation for batch
            LEAST(
                CASE WHEN customer_tier = 'premium' THEN 0.1 ELSE 0.2 END +
                (high_amount * 0.2) +
                (amount_anomaly * 0.3) +
                (CASE WHEN COALESCE(transactions_last_hour, 0) > 5 THEN 0.4 ELSE 0 END),
                0.99
            ) as fraud_prob
        FROM batch_features
    )
    SELECT 
        bp.tx_id,
        bp.fraud_prob,
        CASE 
            WHEN bp.fraud_prob >= 0.8 THEN 'CRITICAL'
            WHEN bp.fraud_prob >= 0.6 THEN 'HIGH'
            WHEN bp.fraud_prob >= 0.3 THEN 'MEDIUM'
            ELSE 'LOW'
        END as risk_lvl,
        EXTRACT(EPOCH FROM (CLOCK_TIMESTAMP() - start_time)) * 1000 / batch_size as avg_proc_time
    FROM batch_predictions bp;
END;
$$;

-- ============================================================================
-- 4. Real-time Feature Extraction
-- ============================================================================

-- Fast feature extraction for real-time predictions
CREATE OR REPLACE FUNCTION ExtractRealTimeFeatures(
    p_customer_id VARCHAR(255),
    p_amount DECIMAL(15,2),
    p_merchant_id VARCHAR(255),
    p_transaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) RETURNS TABLE (
    customer_features JSONB,
    transaction_features JSONB,
    velocity_features JSONB,
    risk_features JSONB
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    WITH customer_data AS (
        SELECT 
            customer_tier,
            account_balance,
            avg_transaction_amount,
            transaction_frequency,
            customer_age_days
        FROM CustomerProfiles 
        WHERE customer_id = p_customer_id
    ),
    velocity_data AS (
        SELECT 
            COALESCE(transactions_last_hour, 0) as tx_hour,
            COALESCE(transactions_last_day, 0) as tx_day,
            COALESCE(amount_last_hour, 0) as amt_hour,
            COALESCE(amount_last_day, 0) as amt_day
        FROM CustomerVelocity 
        WHERE customer_id = p_customer_id
        AND updated_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    ),
    merchant_data AS (
        SELECT 
            risk_category,
            fraud_rate,
            avg_transaction_amount as merchant_avg_amount
        FROM MerchantProfiles 
        WHERE merchant_id = p_merchant_id
    ),
    calculated_features AS (
        SELECT 
            -- Customer features
            JSONB_BUILD_OBJECT(
                'tier', cd.customer_tier,
                'balance', cd.account_balance,
                'avg_amount', cd.avg_transaction_amount,
                'frequency', cd.transaction_frequency,
                'age_days', cd.customer_age_days
            ) as cust_features,
            
            -- Transaction features  
            JSONB_BUILD_OBJECT(
                'amount', p_amount,
                'hour', EXTRACT(HOUR FROM p_transaction_timestamp),
                'day_of_week', EXTRACT(DOW FROM p_transaction_timestamp),
                'month', EXTRACT(MONTH FROM p_transaction_timestamp),
                'merchant_id', p_merchant_id
            ) as tx_features,
            
            -- Velocity features
            JSONB_BUILD_OBJECT(
                'transactions_last_hour', vd.tx_hour,
                'transactions_last_day', vd.tx_day,
                'amount_last_hour', vd.amt_hour,
                'amount_last_day', vd.amt_day
            ) as vel_features,
            
            -- Risk features
            JSONB_BUILD_OBJECT(
                'high_amount', CASE WHEN p_amount > 1000 THEN 1 ELSE 0 END,
                'amount_anomaly', CASE WHEN p_amount > COALESCE(cd.avg_transaction_amount * 3, 500) THEN 1 ELSE 0 END,
                'unusual_time', CASE WHEN EXTRACT(HOUR FROM p_transaction_timestamp) NOT BETWEEN 6 AND 22 THEN 1 ELSE 0 END,
                'high_velocity', CASE WHEN vd.tx_hour > 5 THEN 1 ELSE 0 END,
                'merchant_risk', COALESCE(md.fraud_rate, 0.1),
                'amount_vs_merchant_avg', p_amount / NULLIF(md.merchant_avg_amount, 0)
            ) as risk_features
            
        FROM customer_data cd
        CROSS JOIN velocity_data vd  
        LEFT JOIN merchant_data md ON TRUE
    )
    SELECT 
        cust_features,
        tx_features,
        vel_features,
        risk_features
    FROM calculated_features;
END;
$$;

-- ============================================================================
-- 5. Performance-Optimized Views
-- ============================================================================

-- Materialized view for fast customer lookups
CREATE MATERIALIZED VIEW IF NOT EXISTS FastCustomerLookup AS
SELECT 
    customer_id,
    customer_tier,
    avg_transaction_amount,
    transaction_frequency,
    last_transaction_date,
    total_transactions,
    account_balance,
    risk_score
FROM CustomerProfiles
WHERE last_transaction_date >= CURRENT_DATE - INTERVAL '90 days';

-- Index for fast lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_fast_customer_lookup 
ON FastCustomerLookup(customer_id);

-- Refresh procedure for materialized view
CREATE OR REPLACE PROCEDURE RefreshFastCustomerLookup()
LANGUAGE SQL
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY FastCustomerLookup;
    
    -- Log refresh
    INSERT INTO SystemLog (log_type, message, created_at)
    VALUES ('VIEW_REFRESH', 'FastCustomerLookup refreshed', CURRENT_TIMESTAMP);
END;
$$;

-- ============================================================================
-- 6. Monitoring and Alerting
-- ============================================================================

-- Function to check prediction latency
CREATE OR REPLACE FUNCTION CheckPredictionLatency(
    p_threshold_ms DECIMAL DEFAULT 100.0
) RETURNS TABLE (
    avg_latency_ms DECIMAL(10,4),
    p95_latency_ms DECIMAL(10,4),
    predictions_above_threshold INTEGER,
    total_predictions INTEGER,
    latency_status VARCHAR(20)
)
LANGUAGE SQL
AS $$
BEGIN
    RETURN QUERY
    WITH latency_stats AS (
        SELECT 
            AVG(prediction_latency_ms) as avg_lat,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY prediction_latency_ms) as p95_lat,
            COUNT(*) FILTER (WHERE prediction_latency_ms > p_threshold_ms) as above_threshold,
            COUNT(*) as total_count
        FROM PredictionLog 
        WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
        AND prediction_latency_ms IS NOT NULL
    )
    SELECT 
        avg_lat,
        p95_lat,
        above_threshold::INTEGER,
        total_count::INTEGER,
        CASE 
            WHEN p95_lat <= p_threshold_ms THEN 'GOOD'
            WHEN p95_lat <= p_threshold_ms * 1.5 THEN 'WARNING'
            ELSE 'CRITICAL'
        END::VARCHAR(20)
    FROM latency_stats;
END;
$$;

-- Create alert trigger for high latency
CREATE OR REPLACE FUNCTION AlertHighLatency() 
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.prediction_latency_ms > 200 THEN
        INSERT INTO SystemAlerts (alert_type, message, severity, created_at)
        VALUES (
            'HIGH_LATENCY',
            'Prediction latency exceeded 200ms: ' || NEW.prediction_latency_ms || 'ms',
            'WARNING',
            CURRENT_TIMESTAMP
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_high_latency_alert ON PredictionLog;
CREATE TRIGGER trigger_high_latency_alert
    AFTER INSERT ON PredictionLog
    FOR EACH ROW 
    EXECUTE FUNCTION AlertHighLatency();

-- ============================================================================
-- 7. Supporting Tables for Real-time Predictions
-- ============================================================================

-- Prediction cache table
CREATE TABLE IF NOT EXISTS PredictionCache (
    cache_key VARCHAR(255) PRIMARY KEY,
    result_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

-- Customer velocity tracking
CREATE TABLE IF NOT EXISTS CustomerVelocity (
    customer_id VARCHAR(255) PRIMARY KEY,
    transactions_last_hour INTEGER DEFAULT 0,
    transactions_last_day INTEGER DEFAULT 0,
    amount_last_hour DECIMAL(15,2) DEFAULT 0,
    amount_last_day DECIMAL(15,2) DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Merchant profiles for risk assessment
CREATE TABLE IF NOT EXISTS MerchantProfiles (
    merchant_id VARCHAR(255) PRIMARY KEY,
    merchant_name VARCHAR(255),
    risk_category VARCHAR(50),
    fraud_rate DECIMAL(8,6) DEFAULT 0.1,
    avg_transaction_amount DECIMAL(15,2),
    total_transactions INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System alerts table
CREATE TABLE IF NOT EXISTS SystemAlerts (
    alert_id SERIAL PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'INFO',
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_prediction_cache_created ON PredictionCache(created_at);
CREATE INDEX IF NOT EXISTS idx_customer_velocity_updated ON CustomerVelocity(updated_at);
CREATE INDEX IF NOT EXISTS idx_merchant_profiles_risk ON MerchantProfiles(risk_category);
CREATE INDEX IF NOT EXISTS idx_system_alerts_type ON SystemAlerts(alert_type, created_at);

-- ============================================================================
-- End of Real-time Prediction Queries
-- ============================================================================