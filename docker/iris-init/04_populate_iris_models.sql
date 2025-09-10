-- Complete IRIS IntegratedML Demo Data Setup
-- Creates tables, sample data, and models for dashboard demo

-- ========================================================================
-- Step 1: Create Tables for All 4 Model Demos
-- ========================================================================

-- Credit Risk Demo Table
CREATE TABLE CreditApplications (
    id INTEGER IDENTITY PRIMARY KEY,
    customer_id VARCHAR(50),
    application_date DATE DEFAULT CURRENT_DATE,
    age INTEGER,
    income DECIMAL(12,2),
    credit_score INTEGER,
    loan_amount DECIMAL(12,2),
    employment_length INTEGER,
    default_risk INTEGER,
    confidence DECIMAL(5,4)
);

-- Fraud Detection Demo Table  
CREATE TABLE Transactions (
    id INTEGER IDENTITY PRIMARY KEY,
    transaction_id VARCHAR(50),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    transaction_amount DECIMAL(12,2),
    merchant_category VARCHAR(100),
    card_type VARCHAR(50),
    is_fraud INTEGER,
    fraud_probability DECIMAL(5,4)
);

-- Sales Forecasting Demo Table
CREATE TABLE SalesData (
    id INTEGER IDENTITY PRIMARY KEY,
    date_period DATE,
    product_category VARCHAR(100),
    region VARCHAR(50),
    actual_sales DECIMAL(12,2),
    predicted_sales DECIMAL(12,2),
    seasonality_factor DECIMAL(5,4)
);

-- DNA Similarity Demo Table
CREATE TABLE DNASequences (
    id INTEGER IDENTITY PRIMARY KEY,
    sequence_id VARCHAR(50),
    dna_sequence VARCHAR(1000),
    classification VARCHAR(100),
    similarity_score DECIMAL(5,4),
    gene_family VARCHAR(100),
    organism VARCHAR(100)
);

-- ========================================================================
-- Step 2: Insert Sample Data
-- ========================================================================

-- Credit Risk Sample Data (100 records)
INSERT INTO CreditApplications (customer_id, age, income, credit_score, loan_amount, employment_length, default_risk, confidence)
SELECT 
    'CUST' || LPAD(seq, 6, '0'),
    25 + (seq % 40),
    30000 + (seq * 1000) + ((seq % 7) * 5000),
    550 + (seq % 200),
    5000 + (seq * 500),
    1 + (seq % 15),
    CASE WHEN (seq % 10) < 2 THEN 1 ELSE 0 END,
    0.1 + (seq % 90) / 100.0
FROM (
    SELECT ROW_NUMBER() OVER (ORDER BY %ID) AS seq 
    FROM %SYS.SqlUser.Dummy100
) x;

-- Fraud Detection Sample Data (100 records)
INSERT INTO Transactions (transaction_id, transaction_amount, merchant_category, card_type, is_fraud, fraud_probability)
SELECT 
    'TXN' || LPAD(seq, 8, '0'),
    10.0 + (seq * 15.75),
    CASE (seq % 5) 
        WHEN 0 THEN 'Grocery' 
        WHEN 1 THEN 'Gas Station'
        WHEN 2 THEN 'Restaurant' 
        WHEN 3 THEN 'Online Shopping'
        ELSE 'ATM' 
    END,
    CASE (seq % 3) WHEN 0 THEN 'Visa' WHEN 1 THEN 'MasterCard' ELSE 'Amex' END,
    CASE WHEN (seq % 20) = 0 THEN 1 ELSE 0 END,
    (seq % 95) / 100.0
FROM (
    SELECT ROW_NUMBER() OVER (ORDER BY %ID) AS seq 
    FROM %SYS.SqlUser.Dummy100
) x;

-- Sales Forecasting Sample Data (100 records)
INSERT INTO SalesData (date_period, product_category, region, actual_sales, predicted_sales, seasonality_factor)
SELECT 
    DATEADD(day, -seq, CURRENT_DATE),
    CASE (seq % 4) 
        WHEN 0 THEN 'Electronics' 
        WHEN 1 THEN 'Clothing'
        WHEN 2 THEN 'Home & Garden' 
        ELSE 'Books'
    END,
    CASE (seq % 3) WHEN 0 THEN 'North' WHEN 1 THEN 'South' ELSE 'West' END,
    5000 + (seq * 100) + ((seq % 30) * 200),
    4800 + (seq * 105) + ((seq % 25) * 180),
    0.8 + (seq % 40) / 100.0
FROM (
    SELECT ROW_NUMBER() OVER (ORDER BY %ID) AS seq 
    FROM %SYS.SqlUser.Dummy100
) x;

-- DNA Sequences Sample Data (100 records)
INSERT INTO DNASequences (sequence_id, dna_sequence, classification, similarity_score, gene_family, organism)
SELECT 
    'DNA' || LPAD(seq, 6, '0'),
    'ATCGATCG' || REPEAT('CGTA', seq % 10) || 'TTAAGGCC',
    CASE (seq % 5) 
        WHEN 0 THEN 'Oncogene' 
        WHEN 1 THEN 'Tumor Suppressor'
        WHEN 2 THEN 'Housekeeping' 
        WHEN 3 THEN 'Regulatory'
        ELSE 'Structural'
    END,
    0.5 + (seq % 50) / 100.0,
    CASE (seq % 3) WHEN 0 THEN 'Kinase' WHEN 1 THEN 'Transcription Factor' ELSE 'Receptor' END,
    CASE (seq % 4) 
        WHEN 0 THEN 'Homo sapiens' 
        WHEN 1 THEN 'Mus musculus'
        WHEN 2 THEN 'Drosophila'
        ELSE 'C. elegans'
    END
FROM (
    SELECT ROW_NUMBER() OVER (ORDER BY %ID) AS seq 
    FROM %SYS.SqlUser.Dummy100
) x;

-- ========================================================================
-- Step 3: Create IntegratedML Models
-- ========================================================================

-- Credit Risk Model
CREATE MODEL CreditRiskModel 
PREDICTING (default_risk) 
FROM CreditApplications
SELECT age, income, credit_score, loan_amount, employment_length, default_risk;

-- Train the model
TRAIN MODEL CreditRiskModel;

-- Fraud Detection Model
CREATE MODEL FraudDetectionModel 
PREDICTING (is_fraud) 
FROM Transactions
SELECT transaction_amount, merchant_category, card_type, is_fraud;

-- Train the model
TRAIN MODEL FraudDetectionModel;

-- Sales Forecasting Model
CREATE MODEL SalesForecastModel 
PREDICTING (predicted_sales) 
FROM SalesData
SELECT product_category, region, actual_sales, seasonality_factor, predicted_sales;

-- Train the model
TRAIN MODEL SalesForecastModel;

-- DNA Classifier Model
CREATE MODEL DNAClassifierModel 
PREDICTING (classification) 
FROM DNASequences
SELECT dna_sequence, gene_family, organism, classification;

-- Train the model
TRAIN MODEL DNAClassifierModel;

-- ========================================================================
-- Step 4: Test Models
-- ========================================================================

-- Test all models work
SELECT PREDICT(CreditRiskModel) AS risk_score FROM CreditApplications LIMIT 1;
SELECT PREDICT(FraudDetectionModel) AS fraud_prob FROM Transactions LIMIT 1;
SELECT PREDICT(SalesForecastModel) AS sales_pred FROM SalesData LIMIT 1;
SELECT PREDICT(DNAClassifierModel) AS dna_class FROM DNASequences LIMIT 1;

WRITE "All IntegratedML models created and trained successfully!", !