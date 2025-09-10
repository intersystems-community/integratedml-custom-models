#!/usr/bin/env python3
"""
Simple script to check IRIS database tables
"""
import intersystems_iris.dbapi._DBAPI as dbapi

def check_tables():
    try:
        # Connect to IRIS
        connection = dbapi.connect(
            hostname="flexible_model_integration_iris",
            port=1972,
            namespace="USER",
            username="_SYSTEM",
            password="SYS"
        )
        
        cursor = connection.cursor()
        
        print("=== CHECKING IRIS TABLES ===")
        
        # Check all schemas
        print("\n1. All available tables:")
        cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_SCHEMA, TABLE_NAME")
        all_tables = cursor.fetchall()
        for schema, table in all_tables:
            print(f"   {schema}.{table}")
        
        print(f"\nTotal tables found: {len(all_tables)}")
        
        # Check specific tables we're looking for
        target_tables = [
            'credit_risk_data', 'CreditApplications',
            'fraud_detection_data', 'Transactions', 
            'sales_forecasting_data', 'SalesData',
            'dna_similarity_data', 'DNASequences'
        ]
        
        print("\n2. Checking target tables:")
        for table in target_tables:
            cursor.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table}'")
            exists = cursor.fetchone()[0] > 0
            print(f"   {table}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        # Check IntegratedML models
        print("\n3. IntegratedML models:")
        try:
            cursor.execute("SELECT name FROM INFORMATION_SCHEMA.ML_MODELS")
            models = cursor.fetchall()
            for model in models:
                print(f"   Model: {model[0]}")
        except Exception as e:
            print(f"   Could not query models: {e}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Error connecting to IRIS: {e}")

if __name__ == "__main__":
    check_tables()