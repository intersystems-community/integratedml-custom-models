-- Model Evaluation and Performance Analysis
-- Demo 1: Comprehensive evaluation of credit risk models

-- ========================================================================
-- Section 1: Basic Model Performance Metrics
-- ========================================================================

-- Calculate confusion matrix and basic classification metrics
WITH ModelPredictions AS (
    SELECT 
        application_id,
        default_risk as actual_class,
        PREDICT(CreditRiskModel USING *) as predicted_probability,
        PREDICT(CreditRiskModel WITH 'class' USING *) as predicted_class
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
        AND application_date >= DATEADD(month, -3, CURRENT_DATE)
),
ConfusionMatrix AS (
    SELECT 
        SUM(CASE WHEN actual_class = 0 AND predicted_class = 0 THEN 1 ELSE 0 END) as true_negatives,
        SUM(CASE WHEN actual_class = 0 AND predicted_class = 1 THEN 1 ELSE 0 END) as false_positives,
        SUM(CASE WHEN actual_class = 1 AND predicted_class = 0 THEN 1 ELSE 0 END) as false_negatives,
        SUM(CASE WHEN actual_class = 1 AND predicted_class = 1 THEN 1 ELSE 0 END) as true_positives,
        COUNT(*) as total_predictions
    FROM ModelPredictions
)
SELECT 
    'CreditRiskModel' as model_name,
    CURRENT_DATE as evaluation_date,
    
    -- Confusion Matrix
    true_negatives,
    false_positives, 
    false_negatives,
    true_positives,
    total_predictions,
    
    -- Performance Metrics
    ROUND((true_positives + true_negatives) * 100.0 / total_predictions, 2) as accuracy_pct,
    ROUND(true_positives * 100.0 / NULLIF(true_positives + false_negatives, 0), 2) as recall_pct,
    ROUND(true_positives * 100.0 / NULLIF(true_positives + false_positives, 0), 2) as precision_pct,
    ROUND(true_negatives * 100.0 / NULLIF(true_negatives + false_positives, 0), 2) as specificity_pct,
    
    -- F1 Score calculation
    ROUND(2 * (true_positives * 100.0 / NULLIF(true_positives + false_positives, 0)) * 
              (true_positives * 100.0 / NULLIF(true_positives + false_negatives, 0)) /
          NULLIF((true_positives * 100.0 / NULLIF(true_positives + false_positives, 0)) + 
                 (true_positives * 100.0 / NULLIF(true_positives + false_negatives, 0)), 0), 2) as f1_score

FROM ConfusionMatrix;

-- ========================================================================
-- Section 2: ROC Curve and AUC Analysis
-- ========================================================================

-- Generate ROC curve data points for different thresholds
WITH ThresholdAnalysis AS (
    SELECT 
        threshold,
        SUM(CASE WHEN actual_class = 1 AND predicted_probability >= threshold THEN 1 ELSE 0 END) as tp,
        SUM(CASE WHEN actual_class = 0 AND predicted_probability >= threshold THEN 1 ELSE 0 END) as fp,
        SUM(CASE WHEN actual_class = 1 THEN 1 ELSE 0 END) as total_positives,
        SUM(CASE WHEN actual_class = 0 THEN 1 ELSE 0 END) as total_negatives
    FROM (
        SELECT 
            default_risk as actual_class,
            PREDICT(CreditRiskModel USING *) as predicted_probability
        FROM CreditApplications
        WHERE default_risk IS NOT NULL
            AND application_date >= DATEADD(month, -3, CURRENT_DATE)
    ) predictions
    CROSS JOIN (
        SELECT 0.0 as threshold UNION SELECT 0.1 UNION SELECT 0.2 UNION SELECT 0.3 UNION 
        SELECT 0.4 UNION SELECT 0.5 UNION SELECT 0.6 UNION SELECT 0.7 UNION 
        SELECT 0.8 UNION SELECT 0.9 UNION SELECT 1.0
    ) thresholds
    GROUP BY threshold
)
SELECT 
    threshold,
    ROUND(tp * 100.0 / NULLIF(total_positives, 0), 2) as true_positive_rate,
    ROUND(fp * 100.0 / NULLIF(total_negatives, 0), 2) as false_positive_rate,
    ROUND(tp * 100.0 / NULLIF(tp + fp, 0), 2) as precision,
    tp,
    fp,
    total_positives,
    total_negatives
FROM ThresholdAnalysis
ORDER BY threshold;

-- ========================================================================
-- Section 3: Model Comparison Across Different Configurations
-- ========================================================================

-- Compare performance of different model configurations
WITH ModelComparisons AS (
    SELECT 
        application_id,
        default_risk as actual_class,
        
        -- Main model
        PREDICT(CreditRiskModel USING *) as main_probability,
        PREDICT(CreditRiskModel WITH 'class' USING *) as main_prediction,
        
        -- Conservative model
        PREDICT(ConservativeCreditModel USING *) as conservative_probability,
        PREDICT(ConservativeCreditModel WITH 'class' USING *) as conservative_prediction,
        
        -- Fast model
        PREDICT(FastCreditModel USING *) as fast_probability,
        PREDICT(FastCreditModel WITH 'class' USING *) as fast_prediction
        
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
        AND application_date >= DATEADD(month, -3, CURRENT_DATE)
),
ModelMetrics AS (
    SELECT 
        'CreditRiskModel' as model_name,
        AVG(ABS(actual_class - main_probability)) as mean_absolute_error,
        SQRT(AVG(POWER(actual_class - main_probability, 2))) as root_mean_square_error,
        SUM(CASE WHEN actual_class = main_prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
    FROM ModelComparisons
    
    UNION ALL
    
    SELECT 
        'ConservativeCreditModel' as model_name,
        AVG(ABS(actual_class - conservative_probability)) as mean_absolute_error,
        SQRT(AVG(POWER(actual_class - conservative_probability, 2))) as root_mean_square_error,
        SUM(CASE WHEN actual_class = conservative_prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
    FROM ModelComparisons
    
    UNION ALL
    
    SELECT 
        'FastCreditModel' as model_name,
        AVG(ABS(actual_class - fast_probability)) as mean_absolute_error,
        SQRT(AVG(POWER(actual_class - fast_probability, 2))) as root_mean_square_error,
        SUM(CASE WHEN actual_class = fast_prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
    FROM ModelComparisons
)
SELECT 
    model_name,
    ROUND(mean_absolute_error, 4) as mae,
    ROUND(root_mean_square_error, 4) as rmse,
    ROUND(accuracy, 2) as accuracy_pct,
    CURRENT_TIMESTAMP as evaluation_timestamp
FROM ModelMetrics
ORDER BY accuracy DESC;

-- ========================================================================
-- Section 4: Performance by Customer Segments
-- ========================================================================

-- Analyze model performance across different customer segments
SELECT 
    segment_type,
    segment_value,
    COUNT(*) as total_applications,
    SUM(default_risk) as actual_defaults,
    ROUND(AVG(default_risk) * 100, 2) as actual_default_rate_pct,
    ROUND(AVG(PREDICT(CreditRiskModel USING *)) * 100, 2) as predicted_default_rate_pct,
    ROUND(AVG(ABS(default_risk - PREDICT(CreditRiskModel USING *))) * 100, 2) as mean_absolute_error_pct,
    SUM(CASE WHEN default_risk = PREDICT(CreditRiskModel WITH 'class' USING *) THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_pct
FROM (
    -- Age segments
    SELECT 
        'Age Group' as segment_type,
        CASE 
            WHEN age < 25 THEN 'Under 25'
            WHEN age < 35 THEN '25-34'
            WHEN age < 45 THEN '35-44'
            WHEN age < 55 THEN '45-54'
            WHEN age < 65 THEN '55-64'
            ELSE '65+'
        END as segment_value,
        default_risk,
        age, gender, employment_duration, employment_status, job, monthly_income,
        housing, residence_duration, credit_amount, duration, purpose,
        existing_credits, savings_status, checking_status, credit_history,
        num_dependents, telephone, foreign_worker
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
    
    UNION ALL
    
    -- Credit amount segments
    SELECT 
        'Credit Amount' as segment_type,
        CASE 
            WHEN credit_amount < 5000 THEN 'Under $5K'
            WHEN credit_amount < 15000 THEN '$5K-$15K'
            WHEN credit_amount < 30000 THEN '$15K-$30K'
            ELSE 'Over $30K'
        END as segment_value,
        default_risk,
        age, gender, employment_duration, employment_status, job, monthly_income,
        housing, residence_duration, credit_amount, duration, purpose,
        existing_credits, savings_status, checking_status, credit_history,
        num_dependents, telephone, foreign_worker
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
    
    UNION ALL
    
    -- Employment duration segments
    SELECT 
        'Employment Duration' as segment_type,
        CASE 
            WHEN employment_duration = 0 THEN 'Unemployed'
            WHEN employment_duration < 12 THEN 'Less than 1 year'
            WHEN employment_duration < 48 THEN '1-4 years'
            ELSE 'Over 4 years'
        END as segment_value,
        default_risk,
        age, gender, employment_duration, employment_status, job, monthly_income,
        housing, residence_duration, credit_amount, duration, purpose,
        existing_credits, savings_status, checking_status, credit_history,
        num_dependents, telephone, foreign_worker
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
) segmented_data
WHERE application_date >= DATEADD(month, -6, CURRENT_DATE)
GROUP BY segment_type, segment_value
HAVING COUNT(*) >= 10  -- Only include segments with sufficient data
ORDER BY segment_type, accuracy_pct DESC;

-- ========================================================================
-- Section 5: Feature Importance and Model Interpretability
-- ========================================================================

-- Analyze correlation between features and prediction accuracy
WITH FeatureAnalysis AS (
    SELECT 
        application_id,
        default_risk,
        PREDICT(CreditRiskModel USING *) as predicted_risk,
        ABS(default_risk - PREDICT(CreditRiskModel USING *)) as prediction_error,
        
        -- Key features for analysis
        age,
        employment_duration,
        credit_amount,
        duration,
        monthly_income,
        existing_credits,
        
        -- Derived features
        credit_amount / NULLIF(monthly_income * 12, 0) as debt_to_income_ratio,
        credit_amount / NULLIF(duration, 0) as monthly_payment
        
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
        AND application_date >= DATEADD(month, -3, CURRENT_DATE)
        AND monthly_income > 0
)
SELECT 
    'Feature Correlation Analysis' as analysis_type,
    CORR(age, prediction_error) as age_error_correlation,
    CORR(employment_duration, prediction_error) as employment_error_correlation,
    CORR(credit_amount, prediction_error) as credit_amount_error_correlation,
    CORR(debt_to_income_ratio, prediction_error) as debt_ratio_error_correlation,
    CORR(monthly_payment, prediction_error) as monthly_payment_error_correlation,
    COUNT(*) as sample_size
FROM FeatureAnalysis;

-- ========================================================================
-- Section 6: Model Stability and Drift Analysis
-- ========================================================================

-- Monitor model performance over time to detect drift
SELECT 
    DATE_TRUNC('week', application_date) as week_start,
    COUNT(*) as total_applications,
    AVG(default_risk) as actual_default_rate,
    AVG(PREDICT(CreditRiskModel USING *)) as predicted_default_rate,
    ABS(AVG(default_risk) - AVG(PREDICT(CreditRiskModel USING *))) as prediction_bias,
    AVG(ABS(default_risk - PREDICT(CreditRiskModel USING *))) as mean_absolute_error,
    SUM(CASE WHEN default_risk = PREDICT(CreditRiskModel WITH 'class' USING *) THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as weekly_accuracy,
    
    -- Data distribution metrics
    AVG(age) as avg_age,
    AVG(credit_amount) as avg_credit_amount,
    AVG(employment_duration) as avg_employment_duration,
    AVG(monthly_income) as avg_monthly_income
    
FROM CreditApplications
WHERE default_risk IS NOT NULL
    AND application_date >= DATEADD(month, -6, CURRENT_DATE)
GROUP BY DATE_TRUNC('week', application_date)
HAVING COUNT(*) >= 5  -- Only include weeks with sufficient data
ORDER BY week_start DESC;

-- ========================================================================
-- Section 7: Business Impact Analysis
-- ========================================================================

-- Calculate financial impact of model decisions
WITH BusinessImpact AS (
    SELECT 
        application_id,
        credit_amount,
        default_risk as actual_outcome,
        PREDICT(CreditRiskModel WITH 'class' USING *) as predicted_outcome,
        
        -- Decision outcomes
        CASE 
            WHEN default_risk = 1 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 1 
            THEN 'True Positive - Correctly Rejected'
            WHEN default_risk = 0 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 0 
            THEN 'True Negative - Correctly Approved'
            WHEN default_risk = 1 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 0 
            THEN 'False Negative - Incorrectly Approved'
            WHEN default_risk = 0 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 1 
            THEN 'False Positive - Incorrectly Rejected'
        END as decision_outcome,
        
        -- Financial impact calculation
        CASE 
            WHEN default_risk = 1 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 0 
            THEN -credit_amount * 0.8  -- Loss from bad loan approved
            WHEN default_risk = 0 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 1 
            THEN -credit_amount * 0.05  -- Opportunity cost from good loan rejected
            WHEN default_risk = 0 AND PREDICT(CreditRiskModel WITH 'class' USING *) = 0 
            THEN credit_amount * 0.15   -- Profit from good loan approved
            ELSE 0  -- No cost for correctly rejected bad loans
        END as financial_impact
        
    FROM CreditApplications
    WHERE default_risk IS NOT NULL
        AND application_date >= DATEADD(month, -6, CURRENT_DATE)
)
SELECT 
    decision_outcome,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage,
    SUM(credit_amount) as total_credit_amount,
    SUM(financial_impact) as total_financial_impact,
    ROUND(AVG(financial_impact), 2) as avg_impact_per_application
FROM BusinessImpact
GROUP BY decision_outcome

UNION ALL

SELECT 
    'TOTAL BUSINESS IMPACT' as decision_outcome,
    COUNT(*) as count,
    100.0 as percentage,
    SUM(credit_amount) as total_credit_amount,
    SUM(financial_impact) as total_financial_impact,
    ROUND(AVG(financial_impact), 2) as avg_impact_per_application
FROM BusinessImpact;

-- ========================================================================
-- Section 8: Model Monitoring Dashboard Summary
-- ========================================================================

-- Create comprehensive model health dashboard
SELECT 
    'Model Health Dashboard' as dashboard_section,
    CURRENT_TIMESTAMP as generated_at,
    
    -- Performance Summary
    (SELECT COUNT(*) FROM CreditApplications 
     WHERE application_date >= DATEADD(day, -30, CURRENT_DATE)) as applications_last_30_days,
     
    (SELECT ROUND(AVG(PREDICT(CreditRiskModel USING *)) * 100, 2)
     FROM CreditApplications 
     WHERE application_date >= DATEADD(day, -7, CURRENT_DATE)) as avg_risk_score_last_7_days,
     
    (SELECT COUNT(*) FROM CreditApplications 
     WHERE application_date >= DATEADD(day, -7, CURRENT_DATE)
     AND PREDICT(CreditRiskModel USING *) > 0.8) as high_risk_applications_last_7_days,
     
    -- Model accuracy (last month with known outcomes)
    (SELECT ROUND(SUM(CASE WHEN default_risk = PREDICT(CreditRiskModel WITH 'class' USING *) THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
     FROM CreditApplications 
     WHERE default_risk IS NOT NULL 
     AND application_date >= DATEADD(month, -1, CURRENT_DATE)) as accuracy_last_month,
     
    -- Data quality indicators
    (SELECT COUNT(*) FROM CreditApplications 
     WHERE application_date >= DATEADD(day, -7, CURRENT_DATE)
     AND (age IS NULL OR credit_amount IS NULL OR monthly_income IS NULL)) as incomplete_applications_last_7_days,
     
    -- Model version and last training date
    (SELECT MAX(created_date) FROM INFORMATION_SCHEMA.ML_MODELS 
     WHERE model_name = 'CreditRiskModel') as model_last_updated;