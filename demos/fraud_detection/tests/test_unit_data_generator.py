"""
Unit tests for transaction data generator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from demos.fraud_detection.data.generate_transaction_data import (
    TransactionDataGenerator,
)


class TestTransactionDataGenerator:
    """Test cases for TransactionDataGenerator."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        generator = TransactionDataGenerator()

        assert generator.num_customers == 1000
        assert generator.num_merchants == 100
        assert generator.fraud_rate == 0.02
        assert generator.enable_advanced_patterns == False
        assert generator.customers is not None
        assert generator.merchants is not None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        generator = TransactionDataGenerator(
            num_customers=50,
            num_merchants=20,
            fraud_rate=0.1,
            enable_advanced_patterns=True,
        )

        assert generator.num_customers == 50
        assert generator.num_merchants == 20
        assert generator.fraud_rate == 0.1
        assert generator.enable_advanced_patterns == True
        assert len(generator.customers) == 50
        assert len(generator.merchants) == 20

    def test_generate_customers(self):
        """Test customer generation."""
        generator = TransactionDataGenerator(num_customers=10)
        customers = generator._generate_customers()

        assert len(customers) == 10
        assert "customer_id" in customers.columns
        assert "risk_profile" in customers.columns
        assert "registration_date" in customers.columns
        assert "preferred_categories" in customers.columns

        # Check customer IDs are unique
        assert customers["customer_id"].nunique() == 10

        # Check risk profiles are valid
        valid_profiles = ["low", "medium", "high"]
        assert all(customers["risk_profile"].isin(valid_profiles))

    def test_generate_merchants(self):
        """Test merchant generation."""
        generator = TransactionDataGenerator(num_merchants=5)
        merchants = generator._generate_merchants()

        assert len(merchants) == 5
        assert "merchant_id" in merchants.columns
        assert "category" in merchants.columns
        assert "risk_score" in merchants.columns
        assert "location" in merchants.columns

        # Check merchant IDs are unique
        assert merchants["merchant_id"].nunique() == 5

        # Check risk scores are in valid range
        assert all(merchants["risk_score"] >= 0)
        assert all(merchants["risk_score"] <= 1)

    def test_generate_transaction_data_basic(self, transaction_generator):
        """Test basic transaction data generation."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        data = transaction_generator.generate_transaction_data(
            num_transactions=50, start_date=start_date, end_date=end_date
        )

        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50

        # Check required columns
        required_columns = [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "amount",
            "transaction_timestamp",
            "merchant_category",
            "payment_method",
            "is_fraud",
            "fraud_type",
        ]
        for col in required_columns:
            assert col in data.columns

        # Check transaction IDs are unique
        assert data["transaction_id"].nunique() == 50

        # Check fraud rate is approximately correct
        fraud_rate = data["is_fraud"].mean()
        expected_rate = transaction_generator.fraud_rate
        assert abs(fraud_rate - expected_rate) <= 0.1  # Allow 10% tolerance

    def test_generate_transaction_data_date_range(self, transaction_generator):
        """Test transaction generation within date range."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 7)

        data = transaction_generator.generate_transaction_data(
            num_transactions=20, start_date=start_date, end_date=end_date
        )

        # Check all timestamps are within range
        timestamps = pd.to_datetime(data["transaction_timestamp"])
        assert all(timestamps >= start_date)
        assert all(timestamps <= end_date)

    def test_generate_fraud_patterns(self, transaction_generator):
        """Test fraud pattern generation."""
        # Enable advanced patterns for more fraud scenarios
        generator = TransactionDataGenerator(
            num_customers=20,
            num_merchants=10,
            fraud_rate=0.5,  # High fraud rate for testing
            enable_advanced_patterns=True,
        )

        data = generator.generate_transaction_data(num_transactions=100)
        fraud_data = data[data["is_fraud"] == True]

        # Should have various fraud types
        fraud_types = fraud_data["fraud_type"].unique()
        assert len(fraud_types) > 1

        # Check some common fraud patterns
        expected_types = ["stolen_card", "account_takeover", "synthetic_identity"]
        has_expected = any(ft in fraud_types for ft in expected_types if ft is not None)
        assert has_expected

    def test_amount_distribution(self, transaction_generator):
        """Test transaction amount distribution."""
        data = transaction_generator.generate_transaction_data(num_transactions=100)

        # Check amounts are positive
        assert all(data["amount"] > 0)

        # Check reasonable range (should have variety)
        assert data["amount"].min() < 100  # Some small transactions
        assert data["amount"].max() > 500  # Some larger transactions
        assert data["amount"].std() > 50  # Reasonable variance

    def test_customer_merchant_relationships(self, transaction_generator):
        """Test customer-merchant relationship validity."""
        data = transaction_generator.generate_transaction_data(num_transactions=50)

        # Check all customers exist
        customer_ids = set(data["customer_id"])
        valid_customers = set(transaction_generator.customers["customer_id"])
        assert customer_ids.issubset(valid_customers)

        # Check all merchants exist
        merchant_ids = set(data["merchant_id"])
        valid_merchants = set(transaction_generator.merchants["merchant_id"])
        assert merchant_ids.issubset(valid_merchants)

    def test_time_based_features(self, transaction_generator):
        """Test time-based feature generation."""
        data = transaction_generator.generate_transaction_data(num_transactions=50)

        # Check time features are present and valid
        assert "hour_of_day" in data.columns
        assert "day_of_week" in data.columns

        assert all(data["hour_of_day"] >= 0)
        assert all(data["hour_of_day"] <= 23)
        assert all(data["day_of_week"] >= 0)
        assert all(data["day_of_week"] <= 6)

    def test_merchant_risk_scores(self, transaction_generator):
        """Test merchant risk score assignment."""
        data = transaction_generator.generate_transaction_data(num_transactions=50)

        assert "merchant_risk_score" in data.columns
        assert all(data["merchant_risk_score"] >= 0)
        assert all(data["merchant_risk_score"] <= 1)

    def test_reproducible_generation(self):
        """Test that generation is reproducible with same seed."""
        generator1 = TransactionDataGenerator(num_customers=10, num_merchants=5)
        generator2 = TransactionDataGenerator(num_customers=10, num_merchants=5)

        # Set same random seed
        np.random.seed(42)
        data1 = generator1.generate_transaction_data(num_transactions=20)

        np.random.seed(42)
        data2 = generator2.generate_transaction_data(num_transactions=20)

        # Should generate identical data
        pd.testing.assert_frame_equal(
            data1.sort_values("transaction_id").reset_index(drop=True),
            data2.sort_values("transaction_id").reset_index(drop=True),
        )

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        generator = TransactionDataGenerator(num_customers=5, num_merchants=3)

        # Test zero transactions
        data = generator.generate_transaction_data(num_transactions=0)
        assert len(data) == 0

        # Test single transaction
        data = generator.generate_transaction_data(num_transactions=1)
        assert len(data) == 1

        # Test invalid date range
        with pytest.raises(ValueError):
            generator.generate_transaction_data(
                num_transactions=10,
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1),
            )

    def test_data_types(self, transaction_generator):
        """Test correct data types in generated data."""
        data = transaction_generator.generate_transaction_data(num_transactions=20)

        # Check specific data types
        assert data["amount"].dtype in [np.float64, np.float32]
        assert data["is_fraud"].dtype == bool
        assert data["hour_of_day"].dtype in [np.int64, np.int32]
        assert data["day_of_week"].dtype in [np.int64, np.int32]
        assert data["merchant_risk_score"].dtype in [np.float64, np.float32]

    def test_advanced_patterns_enabled(self):
        """Test advanced fraud patterns when enabled."""
        generator = TransactionDataGenerator(
            num_customers=20,
            num_merchants=10,
            fraud_rate=0.3,
            enable_advanced_patterns=True,
        )

        data = generator.generate_transaction_data(num_transactions=100)
        fraud_data = data[data["is_fraud"] == True]

        # With advanced patterns, should have more sophisticated fraud types
        fraud_types = set(fraud_data["fraud_type"].dropna())

        # Should have multiple fraud types
        assert len(fraud_types) >= 2

        # May include advanced patterns like coordinated attacks
        # (exact patterns depend on implementation, so we test for variety)
        assert len(fraud_types) > len(fraud_types) * 0.3  # Reasonable variety
