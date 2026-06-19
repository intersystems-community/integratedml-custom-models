-- ----------------------------------------------------------------------------
-- AI Functions demo — table setup
--
-- Creates the schema and tables that the AI Function models read/write.
-- Mirrors the example tables used in the SingleStore "Introducing AI
-- Functions" article (product_reviews, support_tickets, user_comments,
-- articles, market_research, legal_documents, transactions, customer_metrics)
-- but adapted to InterSystems IRIS SQL.
-- ----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS AIFunctions;

-- ----- Use case: AI_SENTIMENT over reviews -----
DROP TABLE IF EXISTS AIFunctions.ProductReviews;
CREATE TABLE AIFunctions.ProductReviews (
    review_id    INT PRIMARY KEY,
    review_text  VARCHAR(4000) NOT NULL,
    sentiment    VARCHAR(20)
);

-- ----- Use case: AI_CLASSIFY + AI_SENTIMENT + AI_SUMMARIZE on tickets -----
DROP TABLE IF EXISTS AIFunctions.SupportTickets;
CREATE TABLE AIFunctions.SupportTickets (
    ticket_id     INT PRIMARY KEY,
    message_text  VARCHAR(4000) NOT NULL,
    department    VARCHAR(40),
    summary       VARCHAR(1000),
    sentiment     VARCHAR(20),
    priority      VARCHAR(20)
);

-- ----- Use case: AI_TRANSLATE for global comments -----
DROP TABLE IF EXISTS AIFunctions.UserComments;
CREATE TABLE AIFunctions.UserComments (
    comment_id        INT PRIMARY KEY,
    comment_text      VARCHAR(4000) NOT NULL,
    source_lang       VARCHAR(40)   NOT NULL,
    target_lang       VARCHAR(40)   NOT NULL DEFAULT 'English',
    translated_comment VARCHAR(4000)
);

-- ----- Use case: EMBED_TEXT for semantic search -----
DROP TABLE IF EXISTS AIFunctions.Articles;
CREATE TABLE AIFunctions.Articles (
    article_id      INT PRIMARY KEY,
    title           VARCHAR(200),
    article_body    VARCHAR(8000) NOT NULL,
    content_vector  VARCHAR(16000) -- JSON-serialised float vector
);

-- ----- Use case: AI_SUMMARIZE for executive summaries -----
DROP TABLE IF EXISTS AIFunctions.MarketResearch;
CREATE TABLE AIFunctions.MarketResearch (
    report_id          INT PRIMARY KEY,
    full_report        VARCHAR(16000) NOT NULL,
    executive_summary  VARCHAR(2000)
);

-- ----- Use case: AI_EXTRACT for legal contracts -----
DROP TABLE IF EXISTS AIFunctions.LegalDocuments;
CREATE TABLE AIFunctions.LegalDocuments (
    contract_id      INT PRIMARY KEY,
    contract_text    VARCHAR(16000) NOT NULL,
    renewal_date     VARCHAR(40),
    contract_value   VARCHAR(40),
    counterparties   VARCHAR(200)
);

-- ----- Use case: real-time fraud detection over transactions -----
DROP TABLE IF EXISTS AIFunctions.NewTransactions;
CREATE TABLE AIFunctions.NewTransactions (
    transaction_id   INT PRIMARY KEY,
    user_id          INT NOT NULL,
    amount           DECIMAL(12,2),
    description      VARCHAR(2000) NOT NULL,
    txn_timestamp    TIMESTAMP,
    transaction_vector  VARCHAR(16000),
    fraud_assessment    VARCHAR(2000)
);

-- ----- Use case: churn prediction -----
DROP TABLE IF EXISTS AIFunctions.CustomerMetrics;
CREATE TABLE AIFunctions.CustomerMetrics (
    user_id              INT PRIMARY KEY,
    login_freq           INT,
    avg_sentiment        DECIMAL(5,3),
    days_since_purchase  INT,
    plan                 VARCHAR(40),
    churn_prediction     VARCHAR(2000)
);

-- ----- Prompt staging table for AI_COMPLETE -----
DROP TABLE IF EXISTS AIFunctions.ProductPrompts;
CREATE TABLE AIFunctions.ProductPrompts (
    prompt_id      INT PRIMARY KEY,
    prompt         VARCHAR(4000) NOT NULL,
    product_copy   VARCHAR(8000)
);
