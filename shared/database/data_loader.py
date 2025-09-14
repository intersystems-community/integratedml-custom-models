"""
Data Loading Module for IntegratedML Demos

Handles loading demo data into IRIS database tables for all three demo categories.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
from pathlib import Path

from .connection import IRISConnection, get_connection

# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading for all demo categories."""

    def __init__(self, connection: IRISConnection = None):
        """
        Initialize data loader.

        Args:
            connection: IRIS connection instance
        """
        self.conn = connection or get_connection()
        self.data_path = Path(os.getenv("IML_DATA_PATH", "/app/data"))
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load_all_demo_data(self) -> Dict[str, bool]:
        """
        Load data for all demo categories.

        Returns:
            Dictionary with loading results for each demo
        """
        results = {}

        # Load Credit Risk data
        logger.info("Loading Credit Risk demo data...")
        results["credit_risk"] = self.load_credit_risk_data()

        # Load Fraud Detection data
        logger.info("Loading Fraud Detection demo data...")
        results["fraud_detection"] = self.load_fraud_detection_data()

        # Load Sales Forecasting data
        logger.info("Loading Sales Forecasting demo data...")
        results["sales_forecasting"] = self.load_sales_forecasting_data()

        return results

    def load_credit_risk_data(self, num_samples: int = None) -> bool:
        """
        Load Credit Risk demo data.

        Args:
            num_samples: Number of samples to generate

        Returns:
            True if loading successful, False otherwise
        """
        try:
            num_samples = num_samples or int(os.getenv("CREDIT_RISK_SAMPLES", "10000"))

            # Generate synthetic credit risk data
            data = self._generate_credit_risk_data(num_samples)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Clear existing data
            self.conn.execute_query("DELETE FROM CreditRisk.CustomerData")

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                self._insert_credit_risk_batch(batch)

            logger.info(f"Loaded {len(df)} credit risk records")
            return True

        except Exception as e:
            logger.error(f"Failed to load credit risk data: {e}")
            return False

    def load_fraud_detection_data(self, num_samples: int = None) -> bool:
        """
        Load Fraud Detection demo data.

        Args:
            num_samples: Number of samples to generate

        Returns:
            True if loading successful, False otherwise
        """
        try:
            num_samples = num_samples or int(
                os.getenv("FRAUD_DETECTION_SAMPLES", "50000")
            )

            # Generate synthetic fraud detection data
            data = self._generate_fraud_detection_data(num_samples)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Clear existing data
            self.conn.execute_query("DELETE FROM FraudDetection.TransactionData")

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                self._insert_fraud_detection_batch(batch)

            logger.info(f"Loaded {len(df)} fraud detection records")
            return True

        except Exception as e:
            logger.error(f"Failed to load fraud detection data: {e}")
            return False

    def load_sales_forecasting_data(self, num_days: int = None) -> bool:
        """
        Load Sales Forecasting demo data.

        Args:
            num_days: Number of days of data to generate

        Returns:
            True if loading successful, False otherwise
        """
        try:
            num_days = num_days or int(os.getenv("SALES_FORECASTING_DAYS", "365"))

            # Generate synthetic sales forecasting data
            data = self._generate_sales_forecasting_data(num_days)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Clear existing data
            self.conn.execute_query("DELETE FROM SalesForecasting.SalesData")

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                self._insert_sales_forecasting_batch(batch)

            logger.info(f"Loaded {len(df)} sales forecasting records")
            return True

        except Exception as e:
            logger.error(f"Failed to load sales forecasting data: {e}")
            return False

    def _generate_credit_risk_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic credit risk data."""
        data = []

        for i in range(num_samples):
            # Generate correlated features
            age = np.random.normal(40, 12)
            age = max(18, min(80, age))

            # Income correlated with age
            income = np.random.normal(50000 + age * 1000, 25000)
            income = max(20000, income)

            # Credit score with some correlation to income
            credit_score = np.random.normal(650 + (income - 50000) * 0.002, 100)
            credit_score = max(300, min(850, credit_score))

            # Debt-to-income ratio
            debt_to_income = np.random.beta(2, 5)

            # Employment length
            employment_length = max(0, np.random.normal(8, 5))

            # Loan amount
            loan_amount = np.random.normal(25000, 15000)
            loan_amount = max(1000, loan_amount)

            # Calculate default risk based on features
            risk_score = (
                (credit_score - 650) * -0.01
                + (debt_to_income - 0.3) * 2
                + (loan_amount / income - 0.5) * 1.5
                + np.random.normal(0, 0.3)
            )

            default_risk = 1 if risk_score > 0.5 else 0

            data.append(
                {
                    "age": int(age),
                    "income": round(income, 2),
                    "credit_score": int(credit_score),
                    "debt_to_income_ratio": round(debt_to_income, 4),
                    "employment_length": round(employment_length, 1),
                    "loan_amount": round(loan_amount, 2),
                    "loan_purpose": np.random.choice(
                        ["home", "auto", "personal", "business", "education"]
                    ),
                    "home_ownership": np.random.choice(["own", "rent", "mortgage"]),
                    "annual_income": round(income, 2),
                    "verification_status": np.random.choice(
                        ["verified", "not_verified", "source_verified"]
                    ),
                    "default_risk": default_risk,
                }
            )

        return data

    def _generate_fraud_detection_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic fraud detection data."""
        data = []
        fraud_rate = float(os.getenv("FRAUD_DETECTION_FRAUD_RATE", "0.02"))

        # Define customer base
        num_customers = min(num_samples // 10, 10000)

        for i in range(num_samples):
            customer_id = np.random.randint(1, num_customers + 1)

            # Transaction time
            days_ago = np.random.randint(0, 365)
            transaction_time = datetime.now() - timedelta(days=days_ago)
            transaction_time += timedelta(
                hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60)
            )

            # Time features
            hour_of_day = transaction_time.hour
            is_weekend = 1 if transaction_time.weekday() >= 5 else 0

            # Amount based on time and type
            if hour_of_day < 6 or hour_of_day > 22:
                # Suspicious hours
                base_amount = np.random.lognormal(3, 1.5)
            else:
                base_amount = np.random.lognormal(2.5, 1)

            transaction_amount = max(1, base_amount)

            # Transaction features
            transaction_type = np.random.choice(
                ["purchase", "withdrawal", "transfer", "payment"]
            )
            merchant_category = np.random.choice(
                ["grocery", "gas", "restaurant", "retail", "online", "atm", "other"]
            )

            # Location
            location_country = np.random.choice(
                ["US", "CA", "UK", "FR", "DE", "JP", "OTHER"],
                p=[0.7, 0.1, 0.05, 0.05, 0.03, 0.02, 0.05],
            )
            location_city = f"City_{np.random.randint(1, 1000)}"

            # Velocity features
            days_since_last = max(0, np.random.exponential(3))
            transaction_velocity = np.random.exponential(1)

            # Fraud determination
            fraud_indicators = 0
            if transaction_amount > 1000:
                fraud_indicators += 1
            if hour_of_day < 6 or hour_of_day > 22:
                fraud_indicators += 1
            if location_country != "US":
                fraud_indicators += 1
            if days_since_last < 1 and transaction_velocity > 5:
                fraud_indicators += 2

            # Determine fraud with some randomness
            fraud_probability = min(
                0.9, fraud_indicators * 0.15 + np.random.random() * 0.1
            )
            is_fraud = (
                1
                if np.random.random() < fraud_probability
                and np.random.random() < fraud_rate * 10
                else 0
            )

            data.append(
                {
                    "customer_id": customer_id,
                    "transaction_amount": round(transaction_amount, 2),
                    "transaction_type": transaction_type,
                    "merchant_category": merchant_category,
                    "transaction_time": transaction_time,
                    "location_country": location_country,
                    "location_city": location_city,
                    "is_weekend": is_weekend,
                    "hour_of_day": hour_of_day,
                    "days_since_last_transaction": round(days_since_last, 2),
                    "transaction_velocity": round(transaction_velocity, 4),
                    "is_fraud": is_fraud,
                }
            )

        return data

    def _generate_sales_forecasting_data(self, num_days: int) -> List[Dict[str, Any]]:
        """Generate synthetic sales forecasting data."""
        data = []

        # Product configuration
        num_products = int(os.getenv("SALES_FORECASTING_PRODUCTS", "100"))
        products = [
            {
                "id": i,
                "category": np.random.choice(
                    ["Electronics", "Clothing", "Home", "Books", "Sports"]
                ),
            }
            for i in range(1, num_products + 1)
        ]

        # Generate data for each day
        start_date = datetime.now() - timedelta(days=num_days)

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)

            # Seasonal and trend factors
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            trend_factor = 1 + day_offset * 0.001  # Slight upward trend

            # Weekend effect
            is_weekend = current_date.weekday() >= 5
            weekend_factor = 1.3 if is_weekend else 1.0

            # Generate sales for each product on this day
            for product in products:
                # Base sales with product-specific patterns
                base_sales = (
                    np.random.lognormal(6, 1)
                    * seasonal_factor
                    * trend_factor
                    * weekend_factor
                )

                # Category-specific adjustments
                if product["category"] == "Electronics":
                    base_sales *= 1.5
                elif product["category"] == "Clothing":
                    # Seasonal clothing pattern
                    if day_of_year in range(60, 120) or day_of_year in range(240, 300):
                        base_sales *= 1.4

                # Random promotions
                promotion_active = 1 if np.random.random() < 0.1 else 0
                if promotion_active:
                    base_sales *= 1.8

                sales_amount = max(0, base_sales)
                units_sold = max(1, int(sales_amount / np.random.uniform(10, 100)))

                # Season determination
                month = current_date.month
                if month in [12, 1, 2]:
                    season = "Winter"
                elif month in [3, 4, 5]:
                    season = "Spring"
                elif month in [6, 7, 8]:
                    season = "Summer"
                else:
                    season = "Fall"

                data.append(
                    {
                        "date_key": current_date.date(),
                        "product_id": product["id"],
                        "product_category": product["category"],
                        "sales_amount": round(sales_amount, 2),
                        "units_sold": units_sold,
                        "promotion_active": promotion_active,
                        "season": season,
                        "day_of_week": current_date.weekday() + 1,
                        "month_of_year": current_date.month,
                        "year_value": current_date.year,
                    }
                )

        return data

    def _insert_credit_risk_batch(self, batch_df: pd.DataFrame):
        """Insert batch of credit risk data."""
        insert_sql = """
        INSERT INTO CreditRisk.CustomerData 
        (age, income, credit_score, debt_to_income_ratio, employment_length, 
         loan_amount, loan_purpose, home_ownership, annual_income, 
         verification_status, default_risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for _, row in batch_df.iterrows():
            self.conn.execute_query(
                insert_sql,
                {
                    "age": int(row["age"]),
                    "income": float(row["income"]),
                    "credit_score": int(row["credit_score"]),
                    "debt_to_income_ratio": float(row["debt_to_income_ratio"]),
                    "employment_length": float(row["employment_length"]),
                    "loan_amount": float(row["loan_amount"]),
                    "loan_purpose": str(row["loan_purpose"]),
                    "home_ownership": str(row["home_ownership"]),
                    "annual_income": float(row["annual_income"]),
                    "verification_status": str(row["verification_status"]),
                    "default_risk": int(row["default_risk"]),
                },
            )

    def _insert_fraud_detection_batch(self, batch_df: pd.DataFrame):
        """Insert batch of fraud detection data."""
        insert_sql = """
        INSERT INTO FraudDetection.TransactionData 
        (customer_id, transaction_amount, transaction_type, merchant_category,
         transaction_time, location_country, location_city, is_weekend,
         hour_of_day, days_since_last_transaction, transaction_velocity, is_fraud)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for _, row in batch_df.iterrows():
            self.conn.execute_query(
                insert_sql,
                {
                    "customer_id": int(row["customer_id"]),
                    "transaction_amount": float(row["transaction_amount"]),
                    "transaction_type": str(row["transaction_type"]),
                    "merchant_category": str(row["merchant_category"]),
                    "transaction_time": row["transaction_time"],
                    "location_country": str(row["location_country"]),
                    "location_city": str(row["location_city"]),
                    "is_weekend": int(row["is_weekend"]),
                    "hour_of_day": int(row["hour_of_day"]),
                    "days_since_last_transaction": float(
                        row["days_since_last_transaction"]
                    ),
                    "transaction_velocity": float(row["transaction_velocity"]),
                    "is_fraud": int(row["is_fraud"]),
                },
            )

    def _insert_sales_forecasting_batch(self, batch_df: pd.DataFrame):
        """Insert batch of sales forecasting data."""
        insert_sql = """
        INSERT INTO SalesForecasting.SalesData 
        (date_key, product_id, product_category, sales_amount, units_sold,
         promotion_active, season, day_of_week, month_of_year, year_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for _, row in batch_df.iterrows():
            self.conn.execute_query(
                insert_sql,
                {
                    "date_key": row["date_key"],
                    "product_id": int(row["product_id"]),
                    "product_category": str(row["product_category"]),
                    "sales_amount": float(row["sales_amount"]),
                    "units_sold": int(row["units_sold"]),
                    "promotion_active": int(row["promotion_active"]),
                    "season": str(row["season"]),
                    "day_of_week": int(row["day_of_week"]),
                    "month_of_year": int(row["month_of_year"]),
                    "year_value": int(row["year_value"]),
                },
            )

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all loaded data.

        Returns:
            Dictionary with data summaries
        """
        summary = {}

        try:
            # Credit Risk summary
            credit_sql = """
            SELECT 
                COUNT(*) as total_records,
                AVG(credit_score) as avg_credit_score,
                SUM(default_risk) as total_defaults,
                AVG(loan_amount) as avg_loan_amount
            FROM CreditRisk.CustomerData
            """
            result = self.conn.execute_query(credit_sql)
            if result:
                row = result[0]
                summary["credit_risk"] = {
                    "total_records": row[0],
                    "avg_credit_score": round(row[1], 2) if row[1] else 0,
                    "total_defaults": row[2],
                    "avg_loan_amount": round(row[3], 2) if row[3] else 0,
                }

            # Fraud Detection summary
            fraud_sql = """
            SELECT 
                COUNT(*) as total_records,
                SUM(is_fraud) as total_fraud,
                AVG(transaction_amount) as avg_amount,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM FraudDetection.TransactionData
            """
            result = self.conn.execute_query(fraud_sql)
            if result:
                row = result[0]
                summary["fraud_detection"] = {
                    "total_records": row[0],
                    "total_fraud": row[1],
                    "avg_amount": round(row[2], 2) if row[2] else 0,
                    "unique_customers": row[3],
                }

            # Sales Forecasting summary
            sales_sql = """
            SELECT 
                COUNT(*) as total_records,
                SUM(sales_amount) as total_sales,
                AVG(sales_amount) as avg_daily_sales,
                COUNT(DISTINCT product_id) as unique_products
            FROM SalesForecasting.SalesData
            """
            result = self.conn.execute_query(sales_sql)
            if result:
                row = result[0]
                summary["sales_forecasting"] = {
                    "total_records": row[0],
                    "total_sales": round(row[1], 2) if row[1] else 0,
                    "avg_daily_sales": round(row[2], 2) if row[2] else 0,
                    "unique_products": row[3],
                }

        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            summary["error"] = str(e)

        return summary


def load_all_demo_data() -> Dict[str, bool]:
    """
    Main function to load all demo data.

    Returns:
        Dictionary with loading results
    """
    loader = DataLoader()
    return loader.load_all_demo_data()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load all demo data
    results = load_all_demo_data()

    print("Data Loading Results:")
    for demo, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {demo}: {status}")

    # Print data summary
    loader = DataLoader()
    summary = loader.get_data_summary()
    print("\nData Summary:")
    for demo, stats in summary.items():
        if isinstance(stats, dict) and "error" not in stats:
            print(f"\n{demo.replace('_', ' ').title()}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
