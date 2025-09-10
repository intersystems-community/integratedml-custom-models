-- Fresh IRIS IntegratedML Setup
-- Complete setup from scratch with all tables, data, and models

-- Switch to SQLUSER namespace (default for IRIS Community)
-- Enable IntegratedML
SET ^%SYS("SQLML") = 1;

-- Create Credit Risk tables and data
DROP TABLE IF EXISTS CreditApplications;
CREATE TABLE CreditApplications (
    id INTEGER IDENTITY PRIMARY KEY,
    income NUMERIC(12,2),
    credit_score INTEGER,
    loan_amount NUMERIC(12,2),
    credit_history_length INTEGER,
    debt_to_income NUMERIC(5,2),
    employment_years INTEGER,
    default_risk INTEGER
);

-- Insert Credit Risk sample data
INSERT INTO CreditApplications (income, credit_score, loan_amount, credit_history_length, debt_to_income, employment_years, default_risk) VALUES
(75000, 720, 25000, 8, 0.3, 5, 0),
(45000, 650, 35000, 5, 0.6, 3, 1),
(90000, 780, 40000, 12, 0.2, 8, 0),
(35000, 580, 45000, 3, 0.8, 2, 1),
(120000, 800, 30000, 15, 0.1, 10, 0),
(55000, 690, 28000, 6, 0.4, 4, 0),
(30000, 520, 50000, 2, 0.9, 1, 1),
(80000, 740, 35000, 9, 0.25, 6, 0),
(60000, 670, 32000, 7, 0.45, 5, 0),
(40000, 600, 40000, 4, 0.7, 2, 1);

-- Create Fraud Detection tables and data
DROP TABLE IF EXISTS TransactionData;
CREATE TABLE TransactionData (
    id INTEGER IDENTITY PRIMARY KEY,
    amount NUMERIC(12,2),
    merchant_category VARCHAR(50),
    transaction_hour INTEGER,
    day_of_week INTEGER,
    customer_age INTEGER,
    account_balance NUMERIC(12,2),
    is_fraud INTEGER
);

-- Insert Fraud Detection sample data
INSERT INTO TransactionData (amount, merchant_category, transaction_hour, day_of_week, customer_age, account_balance, is_fraud) VALUES
(1250.50, 'grocery', 14, 3, 35, 5000, 0),
(8500.00, 'electronics', 23, 6, 28, 12000, 1),
(45.75, 'gas_station', 8, 1, 42, 3000, 0),
(15000.00, 'jewelry', 2, 7, 55, 8000, 1),
(67.20, 'restaurant', 19, 5, 31, 2500, 0),
(125.30, 'pharmacy', 11, 2, 67, 6000, 0),
(9999.99, 'online', 3, 4, 22, 1000, 1),
(234.50, 'clothing', 15, 6, 39, 4500, 0),
(500.00, 'atm', 1, 0, 29, 800, 1),
(89.95, 'grocery', 10, 3, 44, 7500, 0);

-- Create Sales Forecasting tables and data
DROP TABLE IF EXISTS SalesData;
CREATE TABLE SalesData (
    id INTEGER IDENTITY PRIMARY KEY,
    sale_date DATE,
    product_category VARCHAR(50),
    quantity INTEGER,
    unit_price NUMERIC(10,2),
    customer_segment VARCHAR(30),
    season VARCHAR(20),
    promotion_active INTEGER,
    revenue NUMERIC(12,2)
);

-- Insert Sales Forecasting sample data
INSERT INTO SalesData (sale_date, product_category, quantity, unit_price, customer_segment, season, promotion_active, revenue) VALUES
('2023-01-15', 'electronics', 5, 299.99, 'premium', 'winter', 0, 1499.95),
('2023-02-20', 'clothing', 12, 89.50, 'standard', 'winter', 1, 1074.00),
('2023-03-10', 'home_garden', 8, 156.75, 'premium', 'spring', 0, 1254.00),
('2023-04-05', 'electronics', 15, 199.99, 'budget', 'spring', 1, 2999.85),
('2023-05-18', 'clothing', 20, 45.25, 'standard', 'spring', 0, 905.00),
('2023-06-22', 'sports', 7, 234.80, 'premium', 'summer', 1, 1643.60),
('2023-07-30', 'electronics', 25, 399.99, 'premium', 'summer', 0, 9999.75),
('2023-08-14', 'home_garden', 18, 78.50, 'budget', 'summer', 1, 1413.00),
('2023-09-25', 'clothing', 30, 67.90, 'standard', 'fall', 1, 2037.00),
('2023-10-12', 'sports', 10, 189.99, 'premium', 'fall', 0, 1899.90);

-- Create DNA Similarity tables and data
DROP TABLE IF EXISTS DNASequences;
CREATE TABLE DNASequences (
    id INTEGER IDENTITY PRIMARY KEY,
    sequence VARCHAR(1000),
    gc_content NUMERIC(5,2),
    length_bp INTEGER,
    organism VARCHAR(100),
    sequence_type VARCHAR(50),
    similarity_score NUMERIC(5,2)
);

-- Insert DNA Similarity sample data
INSERT INTO DNASequences (sequence, gc_content, length_bp, organism, sequence_type, similarity_score) VALUES
('ATCGATCGATCGATCG', 50.00, 16, 'E.coli', 'coding', 85.5),
('GCTAGCTAGCTAGCTA', 50.00, 16, 'human', 'non_coding', 72.3),
('TTTTAAAAGGGGCCCC', 50.00, 16, 'mouse', 'regulatory', 68.9),
('ATATATATATATATAT', 0.00, 16, 'yeast', 'repetitive', 45.2),
('GCGCGCGCGCGCGCGC', 100.00, 16, 'plant', 'structural', 91.7),
('ACGTACGTACGTACGT', 50.00, 16, 'virus', 'coding', 88.1),
('TGCATGCATGCATGCA', 50.00, 16, 'bacteria', 'metabolic', 76.4),
('AAAAAAAAAAAAAAA', 0.00, 15, 'synthetic', 'control', 12.5),
('GGGGGGGGGGGGGGGG', 100.00, 16, 'synthetic', 'control', 15.8),
('ACAGTACAGTACAGTA', 43.75, 16, 'human', 'coding', 82.6);

-- Create IntegratedML Models
CREATE MODEL CreditRiskModel PREDICTING (default_risk) FROM CreditApplications;
CREATE MODEL FraudDetectionModel PREDICTING (is_fraud) FROM TransactionData;
CREATE MODEL SalesForecastModel PREDICTING (revenue) FROM SalesData;
CREATE MODEL DNASimilarityModel PREDICTING (similarity_score) FROM DNASequences;

-- Train the models
TRAIN MODEL CreditRiskModel;
TRAIN MODEL FraudDetectionModel;
TRAIN MODEL SalesForecastModel;
TRAIN MODEL DNASimilarityModel;

-- Create a user for Redash access
CREATE USER redash PASSWORD 'redash123!';
GRANT %ALL TO redash;

-- Create summary view for dashboard
CREATE VIEW MLModelSummary AS 
SELECT 
    'CreditRisk' as demo_type,
    COUNT(*) as total_records,
    AVG(income) as avg_income,
    AVG(credit_score) as avg_credit_score
FROM CreditApplications
UNION ALL
SELECT 
    'FraudDetection' as demo_type,
    COUNT(*) as total_records,
    AVG(amount) as avg_amount,
    AVG(customer_age) as avg_age
FROM TransactionData
UNION ALL
SELECT 
    'SalesForecasting' as demo_type,
    COUNT(*) as total_records,
    AVG(revenue) as avg_revenue,
    AVG(quantity) as avg_quantity
FROM SalesData
UNION ALL
SELECT 
    'DNASimilarity' as demo_type,
    COUNT(*) as total_records,
    AVG(gc_content) as avg_gc_content,
    AVG(similarity_score) as avg_similarity
FROM DNASequences;

WRITE "Fresh IRIS IntegratedML setup completed successfully!", !