CREATE TABLE CreditApplications (
    id INTEGER IDENTITY PRIMARY KEY,
    risk_score DECIMAL(5,4),
    confidence DECIMAL(5,4)
);

CREATE TABLE Transactions (
    id INTEGER IDENTITY PRIMARY KEY,
    fraud_probability DECIMAL(5,4),
    transaction_amount DECIMAL(12,2)
);

CREATE TABLE SalesData (
    id INTEGER IDENTITY PRIMARY KEY,
    predicted_sales DECIMAL(12,2),
    date_period DATE
);

CREATE TABLE DNASequences (
    id INTEGER IDENTITY PRIMARY KEY,
    classification VARCHAR(100),
    similarity_score DECIMAL(5,4)
);

INSERT INTO CreditApplications (risk_score, confidence) VALUES 
(0.15, 0.92), (0.34, 0.87), (0.08, 0.95), (0.67, 0.78), (0.23, 0.91),
(0.45, 0.83), (0.12, 0.94), (0.56, 0.81), (0.78, 0.76), (0.29, 0.89);

INSERT INTO Transactions (fraud_probability, transaction_amount) VALUES 
(0.02, 45.67), (0.89, 1250.00), (0.05, 89.12), (0.03, 156.78), (0.91, 2340.50),
(0.01, 67.89), (0.07, 234.56), (0.95, 5600.00), (0.04, 123.45), (0.02, 78.90);

INSERT INTO SalesData (predicted_sales, date_period) VALUES 
(15678.90, '2025-01-15'), (23456.78, '2025-01-16'), (18900.45, '2025-01-17'),
(21234.67, '2025-01-18'), (19876.54, '2025-01-19'), (22345.89, '2025-01-20'),
(17654.32, '2025-01-21'), (24567.89, '2025-01-22'), (20123.45, '2025-01-23'),
(18765.43, '2025-01-24');

INSERT INTO DNASequences (classification, similarity_score) VALUES 
('Oncogene', 0.87), ('Tumor Suppressor', 0.92), ('Housekeeping', 0.78),
('Regulatory', 0.85), ('Structural', 0.91), ('Oncogene', 0.76),
('Tumor Suppressor', 0.89), ('Housekeeping', 0.83), ('Regulatory', 0.94),
('Structural', 0.88);