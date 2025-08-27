-- Credit Risk Assessment Prediction Examples
-- Demo 1: Making predictions with the trained model

-- ========================================================================
-- Example 1: Basic Risk Assessment for New Applications
-- ========================================================================

-- Score all pending applications from today
SELECT
    application_id,
    customer_id,
    credit_amount,
    duration,
    purpose,
    PREDICT(CreditRiskModel USING *) as risk_probability,
    PREDICT(CreditRiskModel WITH 'class' USING *) as risk_decision,
    CASE
        WHEN PREDICT(CreditRiskModel USING *) >= 0.7 THEN 'REJECT'
        WHEN PREDICT(CreditRiskModel USING *) >= 0.3 THEN 'REVIEW'
        ELSE 'APPROVE'
    END as recommendation
FROM NewCreditApplications
WHERE application_date >= CURRENT_DATE
    AND application_status = 'PENDING'
ORDER BY risk_probability DESC;

-- ========================================================================
-- Example 2: Detailed Risk Analysis with Multiple Models
-- ========================================================================

-- Compare predictions from different model configurations
SELECT
    application_id,
    customer_id,
    credit_amount,
    employment_duration,
    age,
    
    -- Main model predictions
    PREDICT(CreditRiskModel USING *) as main_risk_score,
    PREDICT(CreditRiskModel WITH 'class' USING *) as main_risk_class,
    
    -- Conservative model predictions
    PREDICT(ConservativeCreditModel USING *) as conservative_risk_score,
    PREDICT(ConservativeCreditModel WITH 'class' USING *) as conservative_risk_class,
    
    -- Fast model predictions
    PREDICT(FastCreditModel USING *) as fast_risk_score,
    PREDICT(FastCreditModel WITH 'class' USING *) as fast_risk_class,
    
    -- Model agreement indicator
    CASE
        WHEN PREDICT(CreditRiskModel WITH 'class' USING *) =
             PREDICT(ConservativeCreditModel WITH 'class' USING *) AND
             PREDICT(CreditRiskModel WITH 'class' USING *) =
             PREDICT(FastCreditModel WITH 'class' USING *)
        THEN 'UNANIMOUS'
        WHEN ABS(PREDICT(CreditRiskModel USING *) - PREDICT(ConservativeCreditModel USING *)) < 0.1
        THEN 'CONSENSUS'
        ELSE 'DISAGREEMENT'
    END as model_agreement
    
FROM NewCreditApplications
WHERE application_date >= DATEADD(day, -7, CURRENT_DATE)
ORDER BY main_risk_score DESC
LIMIT 20;

-- ========================================================================
-- Example 3: Risk-Based Credit Amount Recommendations
-- ========================================================================

-- Suggest adjusted credit amounts based on risk profiles
SELECT
    application_id,
    customer_id,
    credit_amount as requested_amount,
    monthly_income,
    age,
    employment_duration,
    
    PREDICT(CreditRiskModel USING *) as risk_score,
    
    -- Risk-adjusted credit amount recommendations
    CASE
        WHEN PREDICT(CreditRiskModel USING *) <= 0.2 THEN credit_amount * 1.0  -- Low risk: full amount
        WHEN PREDICT(CreditRiskModel USING *) <= 0.4 THEN credit_amount * 0.8  -- Medium risk: 80%
        WHEN PREDICT(CreditRiskModel USING *) <= 0.6 THEN credit_amount * 0.6  -- Higher risk: 60%
        WHEN PREDICT(CreditRiskModel USING *) <= 0.8 THEN credit_amount * 0.4  -- High risk: 40%
        ELSE 0  -- Very high risk: reject
    END as recommended_amount,
    
    -- Interest rate adjustments
    CASE
        WHEN PREDICT(CreditRiskModel USING *) <= 0.2 THEN 'PRIME_RATE'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.4 THEN 'PRIME_PLUS_1'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.6 THEN 'PRIME_PLUS_2'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.8 THEN 'PRIME_PLUS_3'
        ELSE 'REJECT'
    END as interest_rate_tier,
    
    -- Reasoning for decision
    CASE
        WHEN PREDICT(CreditRiskModel USING *) <= 0.2 THEN 'Excellent credit profile'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.4 THEN 'Good credit profile with minor concerns'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.6 THEN 'Moderate risk requires reduced exposure'
        WHEN PREDICT(CreditRiskModel USING *) <= 0.8 THEN 'High risk requires significant risk mitigation'
        ELSE 'Risk too high for approval'
    END as decision_reasoning

FROM NewCreditApplications
WHERE application_status = 'PENDING'
    AND credit_amount > 0
ORDER BY risk_score;

-- ========================================================================
-- Example 4: Batch Scoring for Monthly Reporting
-- ========================================================================

-- Generate monthly risk summary report
SELECT
    DATE_TRUNC('month', application_date) as application_month,
    purpose,
    COUNT(*) as total_applications,
    AVG(PREDICT(CreditRiskModel USING *)) as average_risk_score,
    AVG(credit_amount) as average_credit_amount,
    
    -- Risk distribution
    SUM(CASE WHEN PREDICT(CreditRiskModel WITH 'class' USING *) = 1 THEN 1 ELSE 0 END) as high_risk_count,
    SUM(CASE WHEN PREDICT(CreditRiskModel WITH 'class' USING *) = 0 THEN 1 ELSE 0 END) as low_risk_count,
    
    -- Risk percentages
    ROUND(SUM(CASE WHEN PREDICT(CreditRiskModel WITH 'class' USING *) = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as high_risk_percentage,
    
    -- Financial exposure
    SUM(CASE WHEN PREDICT(CreditRiskModel WITH 'class' USING *) = 1 THEN credit_amount ELSE 0 END) as high_risk_exposure,
    SUM(credit_amount) as total_exposure

FROM NewCreditApplications
WHERE application_date >= DATEADD(month, -6, CURRENT_DATE)
GROUP BY DATE_TRUNC('month', application_date), purpose
HAVING COUNT(*) >= 5  -- Only show purpose categories with sufficient volume
ORDER BY application_month DESC, high_risk_percentage DESC;

-- ========================================================================
-- Example 5: Real-time Scoring for API Integration
-- ========================================================================

-- Template for real-time application scoring (suitable for API calls)
SELECT
    :application_id as application_id,
    PREDICT(CreditRiskModel USING (
        age => :age,
        gender => :gender,
        employment_duration => :employment_duration,
        employment_status => :employment_status,
        job => :job,
        monthly_income => :monthly_income,
        housing => :housing,
        residence_duration => :residence_duration,
        credit_amount => :credit_amount,
        duration => :duration,
        purpose => :purpose,
        existing_credits => :existing_credits,
        savings_status => :savings_status,
        checking_status => :checking_status,
        credit_history => :credit_history,
        num_dependents => :num_dependents,
        telephone => :telephone,
        foreign_worker => :foreign_worker
    )) as risk_probability,
    
    PREDICT(CreditRiskModel WITH 'class' USING (
        age => :age,
        gender => :gender,
        employment_duration => :employment_duration,
        employment_status => :employment_status,
        job => :job,
        monthly_income => :monthly_income,
        housing => :housing,
        residence_duration => :residence_duration,
        credit_amount => :credit_amount,
        duration => :duration,
        purpose => :purpose,
        existing_credits => :existing_credits,
        savings_status => :savings_status,
        checking_status => :checking_status,
        credit_history => :credit_history,
        num_dependents => :num_dependents,
        telephone => :telephone,
        foreign_worker => :foreign_worker
    )) as risk_class,
    
    CURRENT_TIMESTAMP as prediction_timestamp;

-- ========================================================================
-- Example 6: Risk Monitoring and Alerts
-- ========================================================================

-- Daily risk monitoring report
SELECT
    CURRENT_DATE as report_date,
    
    -- Today's application summary
    COUNT(*) as todays_applications,
    AVG(PREDICT(CreditRiskModel USING *)) as avg_risk_score,
    MAX(PREDICT(CreditRiskModel USING *)) as max_risk_score,
    
    -- Risk alerts
    SUM(CASE WHEN PREDICT(CreditRiskModel USING *) > 0.9 THEN 1 ELSE 0 END) as very_high_risk_count,
    SUM(CASE WHEN credit_amount > 50000 AND PREDICT(CreditRiskModel USING *) > 0.6 THEN 1 ELSE 0 END) as large_amount_high_risk_count,
    
    -- Exposure calculations
    SUM(CASE WHEN PREDICT(CreditRiskModel WITH 'class' USING *) = 1 THEN credit_amount ELSE 0 END) as high_risk_exposure,
    SUM(credit_amount) as total_exposure_today

FROM NewCreditApplications
WHERE application_date = CURRENT_DATE;

-- ========================================================================
-- Example 7: Model Comparison and A/B Testing
-- ========================================================================

-- Compare model performance on recent applications
WITH ModelComparison AS (
    SELECT
        application_id,
        credit_amount,
        
        PREDICT(CreditRiskModel USING *) as main_model_score,
        PREDICT(ConservativeCreditModel USING *) as conservative_model_score,
        PREDICT(FastCreditModel USING *) as fast_model_score,
        
        -- Calculate score differences
        ABS(PREDICT(CreditRiskModel USING *) - PREDICT(ConservativeCreditModel USING *)) as main_vs_conservative_diff,
        ABS(PREDICT(CreditRiskModel USING *) - PREDICT(FastCreditModel USING *)) as main_vs_fast_diff
        
    FROM NewCreditApplications
    WHERE application_date >= DATEADD(day, -30, CURRENT_DATE)
)
SELECT
    'Model Comparison Summary' as report_type,
    COUNT(*) as total_applications,
    AVG(main_model_score) as avg_main_score,
    AVG(conservative_model_score) as avg_conservative_score,
    AVG(fast_model_score) as avg_fast_score,
    AVG(main_vs_conservative_diff) as avg_main_conservative_difference,
    AVG(main_vs_fast_diff) as avg_main_fast_difference,
    
    -- Agreement metrics
    SUM(CASE WHEN main_vs_conservative_diff < 0.1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as main_conservative_agreement_pct,
    SUM(CASE WHEN main_vs_fast_diff < 0.1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as main_fast_agreement_pct

FROM ModelComparison;