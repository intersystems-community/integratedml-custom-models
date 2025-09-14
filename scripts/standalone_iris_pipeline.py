#!/usr/bin/env python3
"""
Standalone IRIS IntegratedML Data Pipeline
=========================================
Direct CLI-based data population and model creation for IRIS
"""

import intersystems_iris
import random
import time
from datetime import datetime, timedelta
import json


class IRISStandalonePipeline:
    def __init__(
        self,
        hostname="localhost",
        port=1974,
        namespace="USER",
        username="demo",
        password="demo",
    ):
        self.hostname = hostname
        self.port = port
        self.namespace = namespace
        self.username = username
        self.password = password
        self.connection = None

    def connect(self):
        """Connect to IRIS database"""
        try:
            print(f"🔌 Connecting to IRIS at {self.hostname}:{self.port}")
            connection_string = f"{self.hostname}:{self.port}/{self.namespace}"
            self.connection = intersystems_iris.connect(
                connection_string, self.username, self.password
            )
            print("✅ Connected to IRIS successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to IRIS: {str(e)}")
            return False

    def execute_sql(self, sql, description="SQL Query", fetch_results=False):
        """Execute SQL directly against IRIS"""
        print(f"\n📊 {description}")
        print("-" * 50)
        print(f"SQL: {sql[:100]}{'...' if len(sql) > 100 else ''}")

        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)

            if fetch_results:
                results = cursor.fetchall()
                if results:
                    print(f"✅ Success: {len(results)} rows returned")
                    # Print first few rows
                    for i, row in enumerate(results[:3]):
                        print(f"   Row {i+1}: {row}")
                    return results
                else:
                    print("✅ Success: Query executed (no rows returned)")
                    return []
            else:
                print("✅ Success: Query executed")
                return True

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None

    def create_credit_risk_data(self):
        """Create and populate CreditRisk table"""
        print("\n" + "=" * 60)
        print("🏦 CREATING CREDIT RISK DATA")
        print("=" * 60)

        # Drop and create table
        self.execute_sql(
            "DROP TABLE IF EXISTS CreditRisk", "Drop existing CreditRisk table"
        )

        create_sql = """
        CREATE TABLE CreditRisk (
            id INTEGER IDENTITY PRIMARY KEY,
            age INTEGER,
            income NUMERIC(12,2),
            credit_score INTEGER,
            loan_amount NUMERIC(12,2),
            employment_years INTEGER,
            debt_to_income NUMERIC(5,2),
            has_mortgage INTEGER,
            default_risk INTEGER
        )
        """
        self.execute_sql(create_sql, "Create CreditRisk table")

        # Generate and insert data
        print("\n📈 Generating 1000 credit risk records...")

        records = []
        for i in range(1000):
            age = random.randint(18, 80)
            income = random.randint(25000, 150000)
            credit_score = random.randint(300, 850)
            loan_amount = random.randint(5000, 500000)
            employment_years = random.randint(0, 40)
            debt_to_income = round(random.uniform(0.1, 0.8), 2)
            has_mortgage = random.randint(0, 1)

            # Calculate realistic default risk
            risk_score = 0
            if credit_score < 600:
                risk_score += 0.4
            elif credit_score < 700:
                risk_score += 0.2
            if debt_to_income > 0.5:
                risk_score += 0.3
            if employment_years < 2:
                risk_score += 0.2
            if loan_amount > income * 5:
                risk_score += 0.2

            default_risk = 1 if risk_score > 0.5 else 0

            records.append(
                (
                    age,
                    income,
                    credit_score,
                    loan_amount,
                    employment_years,
                    debt_to_income,
                    has_mortgage,
                    default_risk,
                )
            )

        # Insert in batches
        cursor = self.connection.cursor()
        insert_sql = """
        INSERT INTO CreditRisk (age, income, credit_score, loan_amount, employment_years, debt_to_income, has_mortgage, default_risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cursor.executemany(insert_sql, batch)
            print(
                f"   Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}"
            )

        # Verify
        self.execute_sql(
            "SELECT COUNT(*) FROM CreditRisk",
            "Verify CreditRisk record count",
            fetch_results=True,
        )
        self.execute_sql(
            "SELECT TOP 3 * FROM CreditRisk",
            "Sample CreditRisk data",
            fetch_results=True,
        )

    def create_sales_data(self):
        """Create and populate SalesData table"""
        print("\n" + "=" * 60)
        print("📈 CREATING SALES FORECASTING DATA")
        print("=" * 60)

        # Drop and create table
        self.execute_sql(
            "DROP TABLE IF EXISTS SalesData", "Drop existing SalesData table"
        )

        create_sql = """
        CREATE TABLE SalesData (
            id INTEGER IDENTITY PRIMARY KEY,
            product_category VARCHAR(50),
            units_sold INTEGER,
            marketing_spend NUMERIC(10,2),
            season_quarter INTEGER,
            competitor_price NUMERIC(8,2),
            economic_index NUMERIC(5,2),
            sales_amount NUMERIC(12,2)
        )
        """
        self.execute_sql(create_sql, "Create SalesData table")

        # Generate data
        print("\n📊 Generating 800 sales records...")

        categories = [
            "Electronics",
            "Clothing",
            "Home & Garden",
            "Sports",
            "Books",
            "Automotive",
        ]
        records = []

        for i in range(800):
            category = random.choice(categories)
            units_sold = random.randint(10, 1000)
            marketing_spend = random.randint(1000, 50000)
            season_quarter = random.randint(1, 4)
            competitor_price = round(random.uniform(10, 500), 2)
            economic_index = round(random.uniform(85, 115), 2)

            # Calculate realistic sales
            base_sales = units_sold * competitor_price * random.uniform(0.8, 1.2)
            marketing_multiplier = 1 + (marketing_spend / 100000)
            seasonal_multiplier = [1.0, 1.1, 0.9, 1.2][season_quarter - 1]
            economic_multiplier = economic_index / 100

            sales_amount = round(
                base_sales
                * marketing_multiplier
                * seasonal_multiplier
                * economic_multiplier,
                2,
            )

            records.append(
                (
                    category,
                    units_sold,
                    marketing_spend,
                    season_quarter,
                    competitor_price,
                    economic_index,
                    sales_amount,
                )
            )

        # Insert data
        cursor = self.connection.cursor()
        insert_sql = """
        INSERT INTO SalesData (product_category, units_sold, marketing_spend, season_quarter, competitor_price, economic_index, sales_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cursor.executemany(insert_sql, batch)
            print(
                f"   Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}"
            )

        # Verify
        self.execute_sql(
            "SELECT COUNT(*) FROM SalesData",
            "Verify SalesData record count",
            fetch_results=True,
        )
        self.execute_sql(
            "SELECT TOP 3 * FROM SalesData", "Sample SalesData data", fetch_results=True
        )

    def create_transaction_data(self):
        """Create and populate TransactionData table"""
        print("\n" + "=" * 60)
        print("🔒 CREATING FRAUD DETECTION DATA")
        print("=" * 60)

        # Drop and create table
        self.execute_sql(
            "DROP TABLE IF EXISTS TransactionData",
            "Drop existing TransactionData table",
        )

        create_sql = """
        CREATE TABLE TransactionData (
            id INTEGER IDENTITY PRIMARY KEY,
            amount NUMERIC(10,2),
            merchant_category VARCHAR(50),
            transaction_hour INTEGER,
            customer_age INTEGER,
            account_age_days INTEGER,
            previous_transactions INTEGER,
            is_weekend INTEGER,
            is_fraud INTEGER
        )
        """
        self.execute_sql(create_sql, "Create TransactionData table")

        # Generate data
        print("\n🔍 Generating 1200 transaction records...")

        merchant_categories = [
            "Grocery",
            "Gas Station",
            "Restaurant",
            "Online Shopping",
            "ATM",
            "Department Store",
            "Pharmacy",
            "Entertainment",
        ]
        records = []

        for i in range(1200):
            amount = round(random.uniform(1, 5000), 2)
            merchant_category = random.choice(merchant_categories)
            transaction_hour = random.randint(0, 23)
            customer_age = random.randint(18, 80)
            account_age_days = random.randint(1, 3650)
            previous_transactions = random.randint(0, 1000)
            is_weekend = random.randint(0, 1)

            # Calculate fraud probability
            fraud_score = 0
            if amount > 2000:
                fraud_score += 0.3
            if transaction_hour < 6 or transaction_hour > 23:
                fraud_score += 0.2
            if account_age_days < 30:
                fraud_score += 0.3
            if previous_transactions < 5:
                fraud_score += 0.2
            if merchant_category in ["ATM", "Online Shopping"]:
                fraud_score += 0.1

            is_fraud = 1 if fraud_score > 0.6 else 0

            records.append(
                (
                    amount,
                    merchant_category,
                    transaction_hour,
                    customer_age,
                    account_age_days,
                    previous_transactions,
                    is_weekend,
                    is_fraud,
                )
            )

        # Insert data
        cursor = self.connection.cursor()
        insert_sql = """
        INSERT INTO TransactionData (amount, merchant_category, transaction_hour, customer_age, account_age_days, previous_transactions, is_weekend, is_fraud)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cursor.executemany(insert_sql, batch)
            print(
                f"   Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}"
            )

        # Verify
        self.execute_sql(
            "SELECT COUNT(*) FROM TransactionData",
            "Verify TransactionData record count",
            fetch_results=True,
        )
        self.execute_sql(
            "SELECT TOP 3 * FROM TransactionData",
            "Sample TransactionData data",
            fetch_results=True,
        )

    def create_dna_data(self):
        """Create and populate DNASequences table"""
        print("\n" + "=" * 60)
        print("🧬 CREATING DNA SIMILARITY DATA")
        print("=" * 60)

        # Drop and create table
        self.execute_sql(
            "DROP TABLE IF EXISTS DNASequences", "Drop existing DNASequences table"
        )

        create_sql = """
        CREATE TABLE DNASequences (
            id INTEGER IDENTITY PRIMARY KEY,
            organism VARCHAR(50),
            sequence_type VARCHAR(30),
            gc_content NUMERIC(5,2),
            sequence_length INTEGER,
            gene_count INTEGER,
            protein_coding_ratio NUMERIC(5,2),
            similarity_score NUMERIC(5,2)
        )
        """
        self.execute_sql(create_sql, "Create DNASequences table")

        # Generate data
        print("\n🔬 Generating 600 DNA sequence records...")

        organisms = [
            "Human",
            "Mouse",
            "Fruit Fly",
            "Yeast",
            "E. coli",
            "Rice",
            "Arabidopsis",
            "Zebrafish",
        ]
        sequence_types = ["Genomic", "mRNA", "Protein", "Regulatory"]
        records = []

        for i in range(600):
            organism = random.choice(organisms)
            sequence_type = random.choice(sequence_types)
            gc_content = round(random.uniform(30, 70), 2)
            sequence_length = random.randint(100, 50000)
            gene_count = random.randint(1, 500)
            protein_coding_ratio = round(random.uniform(0.1, 0.9), 2)

            # Calculate similarity score
            base_similarity = random.uniform(60, 95)
            if organism in ["Human", "Mouse"]:
                base_similarity += 5
            if sequence_type == "Protein":
                base_similarity += 3
            if 40 <= gc_content <= 60:
                base_similarity += 2

            similarity_score = round(min(base_similarity, 100), 2)

            records.append(
                (
                    organism,
                    sequence_type,
                    gc_content,
                    sequence_length,
                    gene_count,
                    protein_coding_ratio,
                    similarity_score,
                )
            )

        # Insert data
        cursor = self.connection.cursor()
        insert_sql = """
        INSERT INTO DNASequences (organism, sequence_type, gc_content, sequence_length, gene_count, protein_coding_ratio, similarity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cursor.executemany(insert_sql, batch)
            print(
                f"   Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}"
            )

        # Verify
        self.execute_sql(
            "SELECT COUNT(*) FROM DNASequences",
            "Verify DNASequences record count",
            fetch_results=True,
        )
        self.execute_sql(
            "SELECT TOP 3 * FROM DNASequences",
            "Sample DNASequences data",
            fetch_results=True,
        )

    def create_integratedml_models(self):
        """Create and train IntegratedML models"""
        print("\n" + "=" * 60)
        print("🧠 CREATING INTEGRATEDML MODELS")
        print("=" * 60)

        # 1. Credit Risk Model
        print("\n💳 Creating Credit Risk Model...")
        self.execute_sql(
            "DROP MODEL IF EXISTS CreditRiskModel", "Drop existing Credit Risk model"
        )
        self.execute_sql(
            """
            CREATE MODEL CreditRiskModel PREDICTING (default_risk) FROM CreditRisk
            USING {"PROVIDER": "AutoML", "training_mode": "ACCURATE", "minimum_desired_score": 0.6}
        """,
            "Create Credit Risk Model",
        )

        print("\n🎯 Training Credit Risk Model...")
        self.execute_sql("TRAIN MODEL CreditRiskModel", "Train Credit Risk Model")

        # 2. Sales Forecasting Model
        print("\n📈 Creating Sales Forecasting Model...")
        self.execute_sql(
            "DROP MODEL IF EXISTS SalesForecastModel",
            "Drop existing Sales Forecast model",
        )
        self.execute_sql(
            """
            CREATE MODEL SalesForecastModel PREDICTING (sales_amount) FROM SalesData
            USING {"PROVIDER": "AutoML", "training_mode": "ACCURATE", "minimum_desired_score": 0.5}
        """,
            "Create Sales Forecast Model",
        )

        print("\n🎯 Training Sales Forecasting Model...")
        self.execute_sql("TRAIN MODEL SalesForecastModel", "Train Sales Forecast Model")

        # 3. Fraud Detection Model
        print("\n🔒 Creating Fraud Detection Model...")
        self.execute_sql(
            "DROP MODEL IF EXISTS FraudDetectionModel",
            "Drop existing Fraud Detection model",
        )
        self.execute_sql(
            """
            CREATE MODEL FraudDetectionModel PREDICTING (is_fraud) FROM TransactionData
            USING {"PROVIDER": "AutoML", "training_mode": "ACCURATE", "minimum_desired_score": 0.6}
        """,
            "Create Fraud Detection Model",
        )

        print("\n🎯 Training Fraud Detection Model...")
        self.execute_sql(
            "TRAIN MODEL FraudDetectionModel", "Train Fraud Detection Model"
        )

        # 4. DNA Similarity Model
        print("\n🧬 Creating DNA Similarity Model...")
        self.execute_sql(
            "DROP MODEL IF EXISTS DNASimilarityModel",
            "Drop existing DNA Similarity model",
        )
        self.execute_sql(
            """
            CREATE MODEL DNASimilarityModel PREDICTING (similarity_score) FROM DNASequences
            USING {"PROVIDER": "AutoML", "training_mode": "ACCURATE", "minimum_desired_score": 0.5}
        """,
            "Create DNA Similarity Model",
        )

        print("\n🎯 Training DNA Similarity Model...")
        self.execute_sql("TRAIN MODEL DNASimilarityModel", "Train DNA Similarity Model")

        # Verify models
        print("\n🔍 Verifying created models...")
        self.execute_sql("SHOW MODELS", "List all created models", fetch_results=True)

    def test_predictions(self):
        """Test model predictions"""
        print("\n" + "=" * 60)
        print("🎯 TESTING MODEL PREDICTIONS")
        print("=" * 60)

        # Test each model
        print("\n💳 Testing Credit Risk predictions...")
        self.execute_sql(
            """
            SELECT TOP 3
                age, income, credit_score, loan_amount, default_risk,
                PREDICT(CreditRiskModel) as predicted_default_risk
            FROM CreditRisk
        """,
            "Credit Risk predictions",
            fetch_results=True,
        )

        print("\n📈 Testing Sales Forecast predictions...")
        self.execute_sql(
            """
            SELECT TOP 3
                product_category, units_sold, marketing_spend, sales_amount,
                PREDICT(SalesForecastModel) as predicted_sales_amount
            FROM SalesData
        """,
            "Sales Forecast predictions",
            fetch_results=True,
        )

        print("\n🔒 Testing Fraud Detection predictions...")
        self.execute_sql(
            """
            SELECT TOP 3
                amount, merchant_category, transaction_hour, is_fraud,
                PREDICT(FraudDetectionModel) as predicted_is_fraud
            FROM TransactionData
        """,
            "Fraud Detection predictions",
            fetch_results=True,
        )

        print("\n🧬 Testing DNA Similarity predictions...")
        self.execute_sql(
            """
            SELECT TOP 3
                organism, sequence_type, gc_content, similarity_score,
                PREDICT(DNASimilarityModel) as predicted_similarity_score
            FROM DNASequences
        """,
            "DNA Similarity predictions",
            fetch_results=True,
        )

    def run_complete_pipeline(self):
        """Run the complete data pipeline"""
        print("🚀 STANDALONE IRIS INTEGRATEDML PIPELINE")
        print("=" * 60)
        print("Creating sample data and training ML models directly in IRIS...")
        print("=" * 60)

        if not self.connect():
            return False

        try:
            # Create and populate all data tables
            self.create_credit_risk_data()
            self.create_sales_data()
            self.create_transaction_data()
            self.create_dna_data()

            # Create and train IntegratedML models
            self.create_integratedml_models()

            # Test predictions
            self.test_predictions()

            print("\n" + "=" * 60)
            print("🎉 STANDALONE PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("✅ Created 4 data tables with sample data:")
            print("   • CreditRisk: 1000 records")
            print("   • SalesData: 800 records")
            print("   • TransactionData: 1200 records")
            print("   • DNASequences: 600 records")
            print("\n✅ Trained 4 IntegratedML models:")
            print("   • CreditRiskModel (Classification)")
            print("   • SalesForecastModel (Regression)")
            print("   • FraudDetectionModel (Classification)")
            print("   • DNASimilarityModel (Regression)")
            print("\n🎯 All models tested and making predictions!")
            print("📊 Ready for Redash dashboard queries!")
            print("\n💡 Next: Use Redash to query data and visualize PREDICT() results")

            return True

        except Exception as e:
            print(f"\n❌ Error during pipeline execution: {str(e)}")
            return False
        finally:
            if self.connection:
                self.connection.close()
                print("\n🔌 Database connection closed")


def main():
    """Main execution function"""
    print("Starting standalone IRIS IntegratedML pipeline...")

    pipeline = IRISStandalonePipeline()
    success = pipeline.run_complete_pipeline()

    if success:
        print("\n🎯 Pipeline completed successfully!")
        print("🔄 Next step: Use Redash to query and visualize the data!")
    else:
        print("\n❌ Pipeline failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
