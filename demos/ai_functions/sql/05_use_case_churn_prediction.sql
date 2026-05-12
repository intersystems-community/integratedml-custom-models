-- ----------------------------------------------------------------------------
-- Use case 3: Predict and reduce customer churn
--
-- Mirrors the SingleStore article's churn use case: aggregate behavioural
-- metrics, derive a sentiment score from support tickets, and feed the
-- combined signal into AI_COMPLETE for an explainable Yes/No prediction.
--
-- Prerequisite models (already trained in 02_create_models.sql):
--   - AISentiment   → label support tickets
--   - AIComplete    → reason about churn given aggregated metrics
-- ----------------------------------------------------------------------------

-- Step 1: Sentiment score per customer derived from their support history.
WITH ticket_sentiment AS (
    SELECT
        st.user_id,
        AVG(CASE PREDICT(AISentiment USE st.message_text)
                WHEN 'positive' THEN  1.0
                WHEN 'neutral'  THEN  0.0
                WHEN 'negative' THEN -1.0
            END) AS support_sentiment
    FROM AIFunctions.SupportTickets st
    GROUP BY st.user_id
)
-- Step 2: Combine behavioural metrics with sentiment score.
, customer_summary AS (
    SELECT
        cm.user_id,
        cm.login_freq,
        COALESCE(ts.support_sentiment, cm.avg_sentiment) AS combined_sentiment,
        cm.days_since_purchase,
        cm.plan
    FROM AIFunctions.CustomerMetrics cm
    LEFT JOIN ticket_sentiment ts ON ts.user_id = cm.user_id
)
-- Step 3: Build the churn prompt and call AI_COMPLETE.
, prompts AS (
    SELECT
        user_id,
        'Act as a churn analyst. Predict if this customer will churn (Yes/No) '
        || 'and give a reason. Data: '
        || 'plan=' || plan || ', '
        || 'login_freq=' || login_freq || ', '
        || 'sentiment=' || combined_sentiment || ', '
        || 'days_since_purchase=' || days_since_purchase || '.' AS prompt
    FROM customer_summary
)
SELECT
    user_id,
    PREDICT(AIComplete USE prompt) AS churn_prediction
FROM prompts
ORDER BY user_id;

-- Persist the prediction so downstream retention workflows can read it.
UPDATE AIFunctions.CustomerMetrics cm
SET churn_prediction = (
    SELECT PREDICT(AIComplete USE
        'Act as a churn analyst. Predict if this customer will churn (Yes/No) '
        || 'and give a reason. Data: '
        || 'plan=' || cm.plan || ', '
        || 'login_freq=' || cm.login_freq || ', '
        || 'sentiment=' || cm.avg_sentiment || ', '
        || 'days_since_purchase=' || cm.days_since_purchase || '.')
    FROM (VALUES (1)) AS one(c1)
);

-- Customers most at risk: zero/low logins, negative sentiment, stale.
SELECT user_id, plan, login_freq, avg_sentiment, days_since_purchase,
       churn_prediction
FROM AIFunctions.CustomerMetrics
WHERE login_freq <= 5
   OR avg_sentiment < -0.3
   OR days_since_purchase > 90
ORDER BY login_freq ASC, avg_sentiment ASC;
