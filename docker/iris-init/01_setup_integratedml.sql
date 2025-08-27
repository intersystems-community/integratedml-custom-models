-- IntegratedML Setup Script
-- This script initializes the IRIS database for IntegratedML functionality

-- Enable IntegratedML
WRITE "Setting up IntegratedML environment...", !

-- Create namespace if it doesn't exist
IF '$$^%SYS("NAMESPACE","USER") {
    DO ##class(%SYS.Namespace).Create("USER", "/opt/irisapp/data/user/")
}

-- Switch to USER namespace
ZN "USER"

-- Enable IntegratedML for this namespace
SET ^%SYS("SQLML") = 1

-- Create schemas for demo data
CREATE SCHEMA IF NOT EXISTS CreditRisk;
CREATE SCHEMA IF NOT EXISTS FraudDetection;
CREATE SCHEMA IF NOT EXISTS SalesForecasting;

-- Create user for demos
CREATE USER demo PASSWORD 'demo';
GRANT ALL ON SCHEMA CreditRisk TO demo;
GRANT ALL ON SCHEMA FraudDetection TO demo;
GRANT ALL ON SCHEMA SalesForecasting TO demo;

-- Create model storage configuration
CREATE TABLE IF NOT EXISTS ModelRegistry (
    id INTEGER IDENTITY PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    demo_category VARCHAR(100) NOT NULL,
    file_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'INACTIVE',
    metadata JSON
);

-- Create index for fast model lookup
CREATE INDEX IF NOT EXISTS idx_model_name ON ModelRegistry(model_name);
CREATE INDEX IF NOT EXISTS idx_demo_category ON ModelRegistry(demo_category);

WRITE "IntegratedML setup completed successfully!", !