-- ----------------------------------------------------------------------------
-- Use case 2: Real-time fraud detection
--
-- Mirrors the SingleStore article's pipeline:
--   AI_EXTRACT → EMBED_TEXT → DOT_PRODUCT → AI_COMPLETE
-- Each step becomes an IntegratedML Custom Model PREDICT call. DOT_PRODUCT
-- is implemented as a UDF helper (see scripts/run_demo.py for a JSON-decode
-- variant if you don't have IRIS VECTOR support enabled).
-- ----------------------------------------------------------------------------

-- 1. Extract a structured merchant category from the free-text description.
DROP MODEL IF EXISTS AIExtractMerchant;
CREATE MODEL AIExtractMerchant
PREDICTING (description)
FROM AIFunctions.NewTransactions
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_extract",
    "iscmodelsdisabled": 1,
    "userparams": {"question": "What is the merchant category?"}
};
TRAIN MODEL AIExtractMerchant;

-- 2. Embed each transaction description for similarity scoring.
--    (Reuses the EmbedText model registered in 02_create_models.sql.)
UPDATE AIFunctions.NewTransactions
SET transaction_vector = PREDICT(EmbedText USE description);

-- 3. Score each transaction with the AI_COMPLETE reasoner.
--    The reasoner sees: merchant category extracted by AI_EXTRACT, the
--    transaction amount, and the timestamp. (Real systems would add a vector
--    similarity term computed from a user-behaviour vector store.)
WITH enriched AS (
    SELECT
        nt.transaction_id,
        nt.user_id,
        nt.amount,
        nt.txn_timestamp,
        PREDICT(AIExtractMerchant USE nt.description) AS merchant_category,
        nt.description
    FROM AIFunctions.NewTransactions nt
),
prompts AS (
    SELECT
        transaction_id,
        'Is this transaction likely fraudulent? Reply Yes or No and give one '
            || 'sentence of reasoning. Signals: '
            || 'amount=' || amount
            || ', time=' || CAST(txn_timestamp AS VARCHAR(40))
            || ', merchant_category=' || merchant_category
            || ', description=' || description AS prompt
    FROM enriched
)
SELECT
    transaction_id,
    PREDICT(AIComplete USE prompt) AS fraud_assessment
FROM prompts
ORDER BY transaction_id;

-- 4. Persist the fraud assessment back into the transactions table for
--    downstream alerting / case management workflows.
UPDATE AIFunctions.NewTransactions nt
SET fraud_assessment = (
    SELECT PREDICT(AIComplete USE
        'Is this transaction likely fraudulent? Reply Yes or No and give one '
        || 'sentence of reasoning. Signals: amount=' || nt.amount
        || ', time=' || CAST(nt.txn_timestamp AS VARCHAR(40))
        || ', description=' || nt.description)
    FROM (VALUES (1)) AS one(c1)
);
