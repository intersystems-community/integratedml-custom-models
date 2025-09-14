"""
Database Setup and Initialization Module

Sets up IRIS database schemas, tables, and IntegratedML configuration
for the flexible model integration demos.
"""

import os
import logging
from typing import Dict, List, Any
from pathlib import Path

from .connection import get_connection, IRISConnection

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """Handles database initialization and schema setup."""

    def __init__(self, connection: IRISConnection = None):
        """
        Initialize database setup.

        Args:
            connection: IRIS connection instance
        """
        self.conn = connection or get_connection()
        self.project_root = Path(__file__).parent.parent.parent

    def setup_database(self) -> bool:
        """
        Complete database setup including schemas, tables, and IntegratedML configuration.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("Starting database setup...")

            # Test connection first
            if not self.conn.test_connection():
                logger.error("Database connection test failed")
                return False

            # Initialize schemas
            self.initialize_schemas()

            # Create base tables
            self.create_base_tables()

            # Setup IntegratedML configuration
            self.setup_integratedml()

            # Create demo-specific tables
            self.create_demo_tables()

            # Initialize model registry
            self.setup_model_registry()

            logger.info("Database setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False

    def initialize_schemas(self):
        """Create database schemas for different demo categories."""
        schemas = ["CreditRisk", "FraudDetection", "SalesForecasting"]

        for schema in schemas:
            try:
                self.conn.execute_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                logger.info(f"Created schema: {schema}")
            except Exception as e:
                logger.warning(f"Schema {schema} creation warning: {e}")

    def create_base_tables(self):
        """Create base tables for the application."""
        # Create tables one by one for better error handling
        tables = [
            """
            CREATE TABLE IF NOT EXISTS ModelRegistry (
                id INTEGER IDENTITY PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL UNIQUE,
                model_type VARCHAR(100) NOT NULL,
                demo_category VARCHAR(100) NOT NULL,
                file_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) DEFAULT 'INACTIVE',
                metadata LONGVARCHAR,
                performance_metrics LONGVARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ModelPredictions (
                id INTEGER IDENTITY PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                input_data LONGVARCHAR,
                prediction_result LONGVARCHAR,
                confidence_score DECIMAL(5,4),
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time_ms INTEGER
            )
            """,
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_model_name ON ModelRegistry(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_demo_category ON ModelRegistry(demo_category)",
            "CREATE INDEX IF NOT EXISTS idx_status ON ModelRegistry(status)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_model ON ModelPredictions(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_time ON ModelPredictions(prediction_time)",
        ]

        # Execute table creation
        for table_sql in tables:
            try:
                self.conn.execute_sql(table_sql)
            except Exception as e:
                logger.warning(f"Table creation warning: {e}")

        # Execute index creation
        for index_sql in indexes:
            try:
                self.conn.execute_sql(index_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

        logger.info("Created base tables")

    def setup_integratedml(self):
        """Configure IntegratedML settings."""
        try:
            # Create MLConfiguration table
            ml_config_table = """
            CREATE TABLE IF NOT EXISTS MLConfiguration (
                config_key VARCHAR(100) PRIMARY KEY,
                config_value VARCHAR(500),
                description VARCHAR(1000),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            self.conn.execute_sql(ml_config_table)

            # Insert default ML configuration values one by one
            config_values = [
                (
                    "default_provider",
                    "PMML",
                    "Default ML provider for model deployment",
                ),
                ("max_training_time", "3600", "Maximum training time in seconds"),
                ("auto_validation", "true", "Enable automatic model validation"),
                ("default_test_split", "0.2", "Default test split ratio"),
                ("model_cache_size", "100", "Maximum number of cached models"),
            ]

            for key, value, desc in config_values:
                try:
                    insert_sql = f"""
                    INSERT INTO MLConfiguration (config_key, config_value, description)
                    VALUES ('{key}', '{value}', '{desc}')
                    """
                    self.conn.execute_sql(insert_sql)
                except Exception as e:
                    # Config value might already exist
                    logger.debug(f"Config insert warning for {key}: {e}")

            logger.info("Configured IntegratedML settings")

        except Exception as e:
            logger.warning(f"IntegratedML configuration warning: {e}")

    def create_demo_tables(self):
        """Create tables specific to each demo."""

        # Credit Risk Demo Tables
        credit_risk_tables = [
            """
            CREATE TABLE IF NOT EXISTS CreditRisk.CustomerData (
                customer_id INTEGER IDENTITY PRIMARY KEY,
                age INTEGER,
                income DECIMAL(12,2),
                credit_score INTEGER,
                debt_to_income_ratio DECIMAL(5,4),
                employment_length INTEGER,
                loan_amount DECIMAL(12,2),
                loan_purpose VARCHAR(50),
                home_ownership VARCHAR(20),
                annual_income DECIMAL(12,2),
                verification_status VARCHAR(20),
                default_risk INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_credit_customer ON CreditRisk.CustomerData(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_credit_score ON CreditRisk.CustomerData(credit_score)",
        ]

        # Fraud Detection Demo Tables
        fraud_detection_tables = [
            """
            CREATE TABLE IF NOT EXISTS FraudDetection.TransactionData (
                transaction_id INTEGER IDENTITY PRIMARY KEY,
                customer_id INTEGER,
                transaction_amount DECIMAL(12,2),
                transaction_type VARCHAR(50),
                merchant_category VARCHAR(100),
                transaction_time TIMESTAMP,
                location_country VARCHAR(50),
                location_city VARCHAR(100),
                is_weekend INTEGER DEFAULT 0,
                hour_of_day INTEGER,
                days_since_last_transaction INTEGER,
                transaction_velocity DECIMAL(8,4),
                is_fraud INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_fraud_transaction ON FraudDetection.TransactionData(transaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_fraud_customer ON FraudDetection.TransactionData(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_fraud_time ON FraudDetection.TransactionData(transaction_time)",
        ]

        # Sales Forecasting Demo Tables
        sales_forecasting_tables = [
            """
            CREATE TABLE IF NOT EXISTS SalesForecasting.SalesData (
                id INTEGER IDENTITY PRIMARY KEY,
                date_key DATE,
                product_id INTEGER,
                product_category VARCHAR(100),
                sales_amount DECIMAL(12,2),
                units_sold INTEGER,
                promotion_active INTEGER DEFAULT 0,
                season VARCHAR(20),
                day_of_week INTEGER,
                month_of_year INTEGER,
                year_value INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_sales_date ON SalesForecasting.SalesData(date_key)",
            "CREATE INDEX IF NOT EXISTS idx_sales_product ON SalesForecasting.SalesData(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_sales_category ON SalesForecasting.SalesData(product_category)",
        ]

        # Execute all table creation scripts
        all_tables = (
            credit_risk_tables + fraud_detection_tables + sales_forecasting_tables
        )

        for table_sql in all_tables:
            try:
                self.conn.execute_sql(table_sql)
            except Exception as e:
                logger.warning(f"Demo table creation warning: {e}")

        logger.info("Created demo-specific tables")

    def setup_model_registry(self):
        """Initialize model registry with predefined models."""
        models_config = [
            {
                "model_name": "CreditRiskClassifier",
                "model_type": "Classification",
                "demo_category": "CreditRisk",
                "status": "INACTIVE",
            },
            {
                "model_name": "FraudDetectionEnsemble",
                "model_type": "Ensemble",
                "demo_category": "FraudDetection",
                "status": "INACTIVE",
            },
            {
                "model_name": "SalesForecastingHybrid",
                "model_type": "Regression",
                "demo_category": "SalesForecasting",
                "status": "INACTIVE",
            },
        ]

        for model in models_config:
            try:
                insert_sql = f"""
                INSERT INTO ModelRegistry
                (model_name, model_type, demo_category, status, updated_at)
                VALUES ('{model['model_name']}', '{model['model_type']}',
                        '{model['demo_category']}', '{model['status']}', CURRENT_TIMESTAMP)
                """
                self.conn.execute_sql(insert_sql)
                logger.info(f"Registered model: {model['model_name']}")
            except Exception as e:
                logger.warning(
                    f"Model registration warning for {model['model_name']}: {e}"
                )

    def verify_setup(self) -> Dict[str, Any]:
        """
        Verify database setup and return status information.

        Returns:
            Dictionary with setup verification results
        """
        verification = {
            "connection": False,
            "schemas": [],
            "tables": [],
            "integratedml": False,
            "models_registered": 0,
            "errors": [],
        }

        try:
            # Test connection
            verification["connection"] = self.conn.test_connection()

            # Check schemas
            try:
                schema_query = """
                SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA
                WHERE SCHEMA_NAME IN ('CREDITRISK', 'FRAUDDETECTION', 'SALESFORECASTING')
                """
                schemas = self.conn.execute_sql(schema_query)
                verification["schemas"] = [
                    row["SCHEMA_NAME"] if isinstance(row, dict) else row[0]
                    for row in schemas
                ]
            except Exception as e:
                verification["errors"].append(f"Schema check error: {e}")

            # Check tables
            try:
                tables_query = """
                SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME IN ('ModelRegistry', 'ModelPredictions', 'MLConfiguration')
                """
                tables = self.conn.execute_sql(tables_query)
                verification["tables"] = [
                    row["TABLE_NAME"] if isinstance(row, dict) else row[0]
                    for row in tables
                ]
            except Exception as e:
                verification["errors"].append(f"Table check error: {e}")

            # Check IntegratedML
            try:
                ml_info = self.conn.get_integratedml_info()
                verification["integratedml"] = ml_info.get("ml_enabled", False)
            except Exception as e:
                verification["errors"].append(f"IntegratedML check error: {e}")

            # Check registered models
            try:
                models_query = "SELECT COUNT(*) as model_count FROM ModelRegistry"
                models_count = self.conn.execute_sql(models_query)
                if models_count:
                    count_value = (
                        models_count[0]["model_count"]
                        if isinstance(models_count[0], dict)
                        else models_count[0][0]
                    )
                    verification["models_registered"] = count_value
            except Exception as e:
                verification["errors"].append(f"Model count check error: {e}")

        except Exception as e:
            verification["errors"].append(str(e))
            logger.error(f"Setup verification error: {e}")

        return verification


def setup_database() -> bool:
    """
    Main function to set up the database.

    Returns:
        True if setup successful, False otherwise
    """
    try:
        setup = DatabaseSetup()
        return setup.setup_database()
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def initialize_schemas() -> bool:
    """
    Initialize database schemas only.

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        setup = DatabaseSetup()
        setup.initialize_schemas()
        return True
    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False


def verify_database_setup() -> Dict[str, Any]:
    """
    Verify current database setup status.

    Returns:
        Dictionary with verification results
    """
    try:
        setup = DatabaseSetup()
        return setup.verify_setup()
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run database setup
    success = setup_database()
    if success:
        print("Database setup completed successfully!")

        # Verify setup
        verification = verify_database_setup()
        print("\nSetup Verification:")
        for key, value in verification.items():
            print(f"  {key}: {value}")
    else:
        print("Database setup failed!")
        exit(1)
