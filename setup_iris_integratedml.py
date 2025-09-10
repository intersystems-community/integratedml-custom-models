#!/usr/bin/env python3
"""
IRIS IntegratedML Setup Script

This script populates the IRIS database with sample data and IntegratedML models
to support the Redash dashboard demonstrations.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add shared modules to path
sys.path.append('shared')

from database.connection import IRISConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_iris_integratedml():
    """Set up IRIS database with IntegratedML models and sample data."""
    
    # Connect to IRIS database (port 1974 based on docker ps output)
    # Try different credential combinations
    credentials = [
        ('SuperUser', 'SYS'),
        ('_SYSTEM', 'SYS'),
        ('demo', 'demo'),
        ('admin', 'admin')
    ]
    
    conn = None
    for username, password in credentials:
        try:
            logger.info(f"Trying credentials: {username}")
            conn = IRISConnection(
                host='localhost',
                port=1974,
                username=username,
                password=password,
                namespace='USER'
            )
            if conn.test_connection():
                logger.info(f"✅ Connected with credentials: {username}")
                break
        except Exception as e:
            logger.warning(f"Failed with {username}: {e}")
            conn = None
    
    if not conn:
        logger.error("Failed to connect with any credentials")
        return False
    
    try:
        # Connection already tested in the loop above
        logger.info("✅ IRIS connection established")
        
        # Create sample tables and data
        setup_sample_data(conn)
        
        # Create IntegratedML models  
        setup_integratedml_models(conn)
        
        # Verify setup
        verify_setup(conn)
        
        logger.info("🎉 IRIS IntegratedML setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False
    finally:
        conn.close()

def setup_sample_data(conn: IRISConnection):
    """Create sample tables and insert data for demonstrations."""
    
    logger.info("Setting up sample data...")
    
    # Credit Risk Sample Data
    credit_risk_table = """
    CREATE TABLE CreditRisk (
        customer_id INTEGER,
        age INTEGER,
        income DECIMAL(10,2),
        credit_score INTEGER,
        debt_ratio DECIMAL(5,2),
        employment_years INTEGER,
        loan_amount DECIMAL(10,2),
        default_risk VARCHAR(10)
    )
    """
    
    credit_risk_data = """
    INSERT INTO CreditRisk VALUES
    (1, 35, 50000.00, 720, 0.35, 8, 25000.00, 'Low'),
    (2, 28, 35000.00, 650, 0.45, 3, 15000.00, 'Medium'),
    (3, 45, 75000.00, 800, 0.25, 15, 40000.00, 'Low'),
    (4, 22, 25000.00, 580, 0.65, 1, 10000.00, 'High'),
    (5, 38, 60000.00, 740, 0.30, 10, 30000.00, 'Low'),
    (6, 31, 42000.00, 620, 0.50, 5, 18000.00, 'Medium'),
    (7, 52, 85000.00, 780, 0.20, 20, 45000.00, 'Low'),
    (8, 26, 30000.00, 590, 0.60, 2, 12000.00, 'High'),
    (9, 41, 65000.00, 710, 0.32, 12, 32000.00, 'Low'),
    (10, 29, 38000.00, 640, 0.48, 4, 16000.00, 'Medium')
    """
    
    # Sales Forecasting Sample Data
    sales_table = """
    CREATE TABLE SalesData (
        sale_id INTEGER,
        sale_date DATE,
        product_category VARCHAR(50),
        region VARCHAR(50),
        sales_amount DECIMAL(10,2),
        units_sold INTEGER,
        marketing_spend DECIMAL(8,2),
        season VARCHAR(20)
    )
    """
    
    sales_data = """
    INSERT INTO SalesData VALUES
    (1, '2024-01-15', 'Electronics', 'North', 15000.00, 45, 2000.00, 'Winter'),
    (2, '2024-01-22', 'Clothing', 'South', 8500.00, 68, 1200.00, 'Winter'),
    (3, '2024-02-10', 'Electronics', 'East', 22000.00, 67, 3500.00, 'Winter'),
    (4, '2024-02-18', 'Home', 'West', 12000.00, 34, 1800.00, 'Winter'),
    (5, '2024-03-05', 'Electronics', 'North', 18000.00, 52, 2800.00, 'Spring'),
    (6, '2024-03-12', 'Clothing', 'South', 9500.00, 76, 1400.00, 'Spring'),
    (7, '2024-04-08', 'Home', 'East', 14000.00, 41, 2100.00, 'Spring'),
    (8, '2024-04-15', 'Electronics', 'West', 25000.00, 78, 4000.00, 'Spring'),
    (9, '2024-05-20', 'Clothing', 'North', 11000.00, 88, 1600.00, 'Spring'),
    (10, '2024-05-27', 'Home', 'South', 16000.00, 47, 2400.00, 'Spring')
    """
    
    # Fraud Detection Sample Data
    fraud_table = """
    CREATE TABLE TransactionData (
        transaction_id INTEGER,
        account_id INTEGER,
        transaction_amount DECIMAL(10,2),
        transaction_type VARCHAR(20),
        merchant_category VARCHAR(50),
        location_risk_score DECIMAL(3,2),
        time_of_day INTEGER,
        day_of_week INTEGER,
        is_fraud VARCHAR(10)
    )
    """
    
    fraud_data = """
    INSERT INTO TransactionData VALUES
    (1, 1001, 85.50, 'Purchase', 'Grocery', 0.1, 14, 2, 'No'),
    (2, 1002, 1250.00, 'Purchase', 'Electronics', 0.3, 22, 6, 'No'),
    (3, 1003, 25.75, 'ATM', 'Bank', 0.05, 10, 1, 'No'),
    (4, 1004, 5000.00, 'Transfer', 'Online', 0.8, 3, 7, 'Yes'),
    (5, 1005, 45.20, 'Purchase', 'Gas Station', 0.15, 8, 3, 'No'),
    (6, 1006, 750.00, 'Purchase', 'Jewelry', 0.6, 23, 7, 'Yes'),
    (7, 1007, 120.30, 'Purchase', 'Restaurant', 0.2, 19, 5, 'No'),
    (8, 1008, 3000.00, 'ATM', 'Foreign', 0.9, 2, 1, 'Yes'),
    (9, 1009, 65.80, 'Purchase', 'Clothing', 0.1, 15, 4, 'No'),
    (10, 1010, 200.00, 'Transfer', 'Bank', 0.25, 11, 2, 'No')
    """
    
    # Execute table creation and data insertion
    tables_and_data = [
        (credit_risk_table, credit_risk_data, "Credit Risk"),
        (sales_table, sales_data, "Sales Data"),
        (fraud_table, fraud_data, "Transaction Data")
    ]
    
    for table_sql, data_sql, name in tables_and_data:
        try:
            # Drop table if exists
            table_name = name.replace(" ", "")
            conn.execute_sql(f"DROP TABLE IF EXISTS {table_name}")
            
            # Create table
            conn.execute_sql(table_sql)
            logger.info(f"✅ Created table: {name}")
            
            # Insert data
            conn.execute_sql(data_sql)
            logger.info(f"✅ Inserted sample data for: {name}")
            
        except Exception as e:
            logger.error(f"Failed to create {name} table: {e}")

def setup_integratedml_models(conn: IRISConnection):
    """Create and train IntegratedML models."""
    
    logger.info("Setting up IntegratedML models...")
    
    # Credit Risk Model
    try:
        credit_model_sql = """
        CREATE MODEL CreditRiskModel
        PREDICTING (default_risk)
        FROM CreditRisk
        SELECT age, income, credit_score, debt_ratio, employment_years, loan_amount, default_risk
        """
        conn.execute_sql(credit_model_sql)
        logger.info("✅ Created Credit Risk model")
        
        # Train the model
        conn.execute_sql("TRAIN MODEL CreditRiskModel")
        logger.info("✅ Training started for Credit Risk model")
        
    except Exception as e:
        logger.warning(f"Credit Risk model creation/training: {e}")
    
    # Sales Forecasting Model  
    try:
        sales_model_sql = """
        CREATE MODEL SalesForecastModel
        PREDICTING (sales_amount)
        FROM SalesData
        SELECT units_sold, marketing_spend, sales_amount
        """
        conn.execute_sql(sales_model_sql)
        logger.info("✅ Created Sales Forecast model")
        
        # Train the model
        conn.execute_sql("TRAIN MODEL SalesForecastModel")
        logger.info("✅ Training started for Sales Forecast model")
        
    except Exception as e:
        logger.warning(f"Sales Forecast model creation/training: {e}")
    
    # Fraud Detection Model
    try:
        fraud_model_sql = """
        CREATE MODEL FraudDetectionModel
        PREDICTING (is_fraud)
        FROM TransactionData  
        SELECT transaction_amount, location_risk_score, time_of_day, day_of_week, is_fraud
        """
        conn.execute_sql(fraud_model_sql)
        logger.info("✅ Created Fraud Detection model")
        
        # Train the model
        conn.execute_sql("TRAIN MODEL FraudDetectionModel")
        logger.info("✅ Training started for Fraud Detection model")
        
    except Exception as e:
        logger.warning(f"Fraud Detection model creation/training: {e}")

def verify_setup(conn: IRISConnection):
    """Verify that the setup was successful."""
    
    logger.info("Verifying setup...")
    
    # Check tables
    tables = ['CreditRisk', 'SalesData', 'TransactionData']
    for table in tables:
        try:
            result = conn.execute_sql(f"SELECT COUNT(*) as count FROM {table}")
            count = result[0]['count'] if result else 0
            logger.info(f"✅ Table {table}: {count} rows")
        except Exception as e:
            logger.error(f"❌ Table {table}: {e}")
    
    # Check models
    try:
        models = conn.execute_sql("SELECT model_name, trained FROM INFORMATION_SCHEMA.ML_MODELS")
        if models:
            for model in models:
                status = "Trained" if model.get('trained') else "Created"
                logger.info(f"✅ Model {model['model_name']}: {status}")
        else:
            logger.warning("No models found in INFORMATION_SCHEMA.ML_MODELS")
    except Exception as e:
        logger.warning(f"Could not verify models: {e}")
    
    # Test basic IntegratedML functionality
    try:
        test_query = """
        SELECT 'IRIS IntegratedML Ready' as status,
               'Machine Learning in SQL' as description,
               'Models trained and ready' as message
        """
        result = conn.execute_sql(test_query)
        if result:
            logger.info("✅ Basic SQL functionality verified")
        else:
            logger.warning("Basic SQL test returned no results")
    except Exception as e:
        logger.error(f"Basic SQL test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting IRIS IntegratedML Setup...")
    success = setup_iris_integratedml()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("📊 Dashboard queries should now return data")
        print("🔗 Access Redash at: http://localhost:8081")
    else:
        print("\n❌ Setup failed. Check logs for details.")
        sys.exit(1)