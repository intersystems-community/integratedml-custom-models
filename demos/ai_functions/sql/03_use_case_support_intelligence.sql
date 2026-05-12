-- ----------------------------------------------------------------------------
-- Use case 1: Automate customer support intelligence
--
-- Mirrors the SingleStore article's first use case:
--   SELECT
--     ticket_id,
--     aura.AI_SENTIMENT(message_text) as sentiment,
--     aura.AI_CLASSIFY(message_text, '["billing","technical","returns"]')
--       as department,
--     aura.AI_SUMMARIZE(message_text, 10) as summary
--   FROM support_tickets WHERE processed = FALSE;
--
-- IRIS IntegratedML version: each `aura.AI_*` call becomes a `PREDICT(<model>)`
-- against the corresponding Custom Model trained in 02_create_models.sql.
-- Models read the input text from the FROM clause's columns, so we wrap them
-- with subqueries when we need to feed a different table.
-- ----------------------------------------------------------------------------

-- Process all unhandled tickets in a single SQL statement.
SELECT
    ticket_id,
    PREDICT(AISentiment USE message_text) AS sentiment,
    PREDICT(AIClassify  USE message_text) AS department,
    PREDICT(AISummarize USE message_text) AS summary
FROM AIFunctions.SupportTickets
ORDER BY ticket_id;

-- Push the enriched rows back into the table for downstream use.
UPDATE AIFunctions.SupportTickets
SET sentiment  = PREDICT(AISentiment USE message_text),
    department = PREDICT(AIClassify  USE message_text),
    summary    = PREDICT(AISummarize USE message_text);

-- Escalation queue: highly negative sentiment regardless of department.
SELECT
    ticket_id,
    department,
    sentiment,
    summary,
    'P1 — escalate to senior agent' AS routing_action
FROM AIFunctions.SupportTickets
WHERE sentiment = 'negative'
ORDER BY ticket_id;

-- Aggregate view for the support-ops dashboard.
SELECT
    department,
    COUNT(*)                                                  AS total_tickets,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END)   AS positive,
    SUM(CASE WHEN sentiment = 'neutral'  THEN 1 ELSE 0 END)   AS neutral,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END)   AS negative
FROM AIFunctions.SupportTickets
GROUP BY department
ORDER BY total_tickets DESC;
