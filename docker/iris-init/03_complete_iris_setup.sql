-- Drop existing tables if they exist
DROP TABLE IF EXISTS CreditApplications;
DROP TABLE IF EXISTS Transactions;
DROP TABLE IF EXISTS SalesData;
DROP TABLE IF EXISTS DNASequences;

-- Create tables with proper input features for IntegratedML training
CREATE TABLE CreditApplications (
    id INTEGER IDENTITY PRIMARY KEY,
    income DECIMAL(12,2),
    credit_score INTEGER,
    loan_amount DECIMAL(12,2),
    default_risk DECIMAL(5,4)
);

CREATE TABLE Transactions (
    id INTEGER IDENTITY PRIMARY KEY,
    transaction_amount DECIMAL(12,2),
    merchant_category VARCHAR(50),
    is_fraud INTEGER
);

CREATE TABLE SalesData (
    id INTEGER IDENTITY PRIMARY KEY,
    date_period DATE,
    product_category VARCHAR(50),
    actual_sales DECIMAL(12,2)
);

CREATE TABLE DNASequences (
    id INTEGER IDENTITY PRIMARY KEY,
    sequence_length INTEGER,
    gc_content DECIMAL(5,4),
    classification VARCHAR(100)
);

-- Insert sample data for Credit Applications
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (50000, 680, 15000, 0.15);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (75000, 720, 25000, 0.08);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (40000, 620, 12000, 0.34);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (90000, 780, 35000, 0.05);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (35000, 580, 8000, 0.67);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (65000, 700, 20000, 0.12);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (80000, 750, 30000, 0.06);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (45000, 640, 16000, 0.23);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (100000, 800, 40000, 0.03);
INSERT INTO CreditApplications (income, credit_score, loan_amount, default_risk) VALUES (55000, 660, 18000, 0.18);

-- Insert sample data for Transactions
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (45.67, 'grocery', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (1250.00, 'electronics', 1);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (89.12, 'restaurant', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (156.78, 'gas_station', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (2340.50, 'online', 1);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (67.89, 'pharmacy', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (234.56, 'clothing', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (5600.00, 'jewelry', 1);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (123.45, 'grocery', 0);
INSERT INTO Transactions (transaction_amount, merchant_category, is_fraud) VALUES (78.90, 'restaurant', 0);

-- Insert sample data for Sales Data
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-15', 'electronics', 15678.90);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-16', 'clothing', 23456.78);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-17', 'home_garden', 18900.45);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-18', 'electronics', 21234.67);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-19', 'books', 19876.54);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-20', 'clothing', 22345.89);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-21', 'home_garden', 17654.32);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-22', 'electronics', 24567.89);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-23', 'books', 20123.45);
INSERT INTO SalesData (date_period, product_category, actual_sales) VALUES ('2025-01-24', 'clothing', 18765.43);

-- Insert sample data for DNA Sequences
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (1250, 0.45, 'Oncogene');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (890, 0.62, 'Tumor Suppressor');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (2100, 0.38, 'Housekeeping');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (1560, 0.51, 'Regulatory');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (750, 0.67, 'Structural');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (1820, 0.42, 'Oncogene');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (1100, 0.58, 'Tumor Suppressor');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (2350, 0.35, 'Housekeeping');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (1430, 0.49, 'Regulatory');
INSERT INTO DNASequences (sequence_length, gc_content, classification) VALUES (980, 0.64, 'Structural');