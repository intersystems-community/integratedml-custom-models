import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

MERCHANT_CATEGORIES = [
    "grocery",
    "gas_station",
    "restaurant",
    "retail",
    "online",
    "entertainment",
    "travel",
    "cash_advance",
]

FRAUD_TYPES = [
    "stolen_card",
    "account_takeover",
    "synthetic_identity",
    "card_testing",
    "velocity_fraud",
]

PAYMENT_METHODS = ["credit_card", "debit_card", "prepaid_card", "digital_wallet"]


class TransactionDataGenerator:

    def __init__(
        self,
        num_customers: int = 1000,
        num_merchants: int = 100,
        fraud_rate: float = 0.02,
        enable_advanced_patterns: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.fraud_rate = fraud_rate
        self.enable_advanced_patterns = enable_advanced_patterns
        self.random_state = random_state

        np.random.seed(random_state)
        self.customers = self._generate_customers()
        self.merchants = self._generate_merchants()

    def _generate_customers(self) -> pd.DataFrame:
        n = self.num_customers
        risk_profiles = np.random.choice(["low", "medium", "high"], size=n, p=[0.6, 0.3, 0.1])
        customer_ids = [f"CUST_{i:06d}" for i in range(n)]
        reg_dates = [
            datetime(2020, 1, 1) + timedelta(days=int(d))
            for d in np.random.randint(0, 1460, size=n)
        ]
        preferred = [
            ",".join(np.random.choice(MERCHANT_CATEGORIES, size=3, replace=False).tolist())
            for _ in range(n)
        ]
        return pd.DataFrame(
            {
                "customer_id": customer_ids,
                "risk_profile": risk_profiles,
                "registration_date": reg_dates,
                "preferred_categories": preferred,
            }
        )

    def _generate_merchants(self) -> pd.DataFrame:
        n = self.num_merchants
        merchant_ids = [f"MERCH_{i:05d}" for i in range(n)]
        categories = np.random.choice(MERCHANT_CATEGORIES, size=n)
        risk_scores = np.random.uniform(0.0, 1.0, size=n).round(4)
        locations = [f"{np.random.uniform(25, 49):.4f},{np.random.uniform(-125, -66):.4f}" for _ in range(n)]
        return pd.DataFrame(
            {
                "merchant_id": merchant_ids,
                "category": categories,
                "risk_score": risk_scores,
                "location": locations,
            }
        )

    def generate_transaction_data(
        self,
        num_transactions: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        start_date = start_date or datetime(2024, 1, 1)
        end_date = end_date or datetime(2024, 3, 31)

        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")

        if num_transactions == 0:
            return pd.DataFrame(
                columns=[
                    "transaction_id",
                    "customer_id",
                    "merchant_id",
                    "amount",
                    "transaction_timestamp",
                    "merchant_category",
                    "payment_method",
                    "is_fraud",
                    "fraud_type",
                    "hour_of_day",
                    "day_of_week",
                    "merchant_risk_score",
                ]
            )

        n_fraud = max(0, int(num_transactions * self.fraud_rate))

        total_seconds = max(1, int((end_date - start_date).total_seconds()))
        offsets = np.random.randint(0, total_seconds, size=num_transactions)
        timestamps = [start_date + timedelta(seconds=int(s)) for s in offsets]

        customer_sample = self.customers["customer_id"].values
        chosen_customers = np.random.choice(customer_sample, size=num_transactions)

        merchant_ids = self.merchants["merchant_id"].values
        merchant_categories = self.merchants["category"].values
        merchant_risks = self.merchants["risk_score"].values

        chosen_merchant_idx = np.random.randint(0, len(merchant_ids), size=num_transactions)
        chosen_merchant_ids = merchant_ids[chosen_merchant_idx]
        chosen_categories = merchant_categories[chosen_merchant_idx]
        chosen_risks = merchant_risks[chosen_merchant_idx]

        amounts = np.abs(np.random.lognormal(mean=4.5, sigma=1.0, size=num_transactions)).round(2)

        payment_methods = np.random.choice(PAYMENT_METHODS, size=num_transactions)

        is_fraud = np.zeros(num_transactions, dtype=bool)
        is_fraud[:n_fraud] = True
        shuffle_idx = np.random.permutation(num_transactions)
        is_fraud = is_fraud[shuffle_idx]

        fraud_type = np.full(num_transactions, None, dtype=object)
        if n_fraud > 0:
            fraud_type[np.where(is_fraud)[0]] = np.random.choice(FRAUD_TYPES, size=n_fraud)

        hours = np.array([t.hour for t in timestamps], dtype=np.int64)
        days = np.array([t.weekday() for t in timestamps], dtype=np.int64)

        transaction_ids = [f"TXN_{i:010d}" for i in range(num_transactions)]

        df = pd.DataFrame(
            {
                "transaction_id": transaction_ids,
                "customer_id": chosen_customers,
                "merchant_id": chosen_merchant_ids,
                "amount": amounts,
                "transaction_timestamp": timestamps,
                "merchant_category": chosen_categories,
                "payment_method": payment_methods,
                "is_fraud": is_fraud,
                "fraud_type": fraud_type,
                "hour_of_day": hours,
                "day_of_week": days,
                "merchant_risk_score": chosen_risks,
            }
        )

        return df

    def generate_transactions(self, n_transactions=None, num_transactions=None, **kwargs):
        if num_transactions is None:
            num_transactions = n_transactions
        return self.generate_transaction_data(num_transactions=num_transactions, **kwargs)
