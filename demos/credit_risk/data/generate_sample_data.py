"""
Sample Credit Risk Dataset Generator

This module generates realistic synthetic credit application data for the
IntegratedML Credit Risk Assessment demo. The data follows the structure
of the German Credit dataset but with additional synthetic features to
demonstrate custom feature engineering capabilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CreditDataGenerator:
    """
    Generator for realistic synthetic credit application data.

    Creates data that mimics real-world credit applications with:
    - Customer demographics (age, employment, residence)
    - Financial information (income, existing credits, savings)
    - Credit request details (amount, duration, purpose)
    - Credit history and risk indicators
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the credit data generator.

        Parameters:
        -----------
        random_seed : int
            Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define realistic value ranges and distributions
        self._setup_distributions()

    def _reset_random_state(self):
        """
        Reset random state to ensure reproducible data generation.
        This method should be called before each dataset generation.
        """
        logger.debug(
            f"Resetting random seed to {self.random_seed} for reproducible generation"
        )
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _setup_distributions(self):
        """Setup realistic distributions for various features."""

        # Age distribution (weighted toward working age)
        self.age_weights = {
            (18, 25): 0.15,  # Young adults
            (25, 35): 0.25,  # Young professionals
            (35, 45): 0.25,  # Mid-career
            (45, 55): 0.20,  # Senior professionals
            (55, 65): 0.10,  # Pre-retirement
            (65, 75): 0.05,  # Seniors
        }

        # Employment duration (months)
        self.employment_patterns = {
            "unemployed": {"duration": 0, "weight": 0.05},
            "less_than_1_year": {"duration": (1, 12), "weight": 0.15},
            "1_to_4_years": {"duration": (12, 48), "weight": 0.30},
            "4_to_7_years": {"duration": (48, 84), "weight": 0.25},
            "more_than_7_years": {"duration": (84, 240), "weight": 0.25},
        }

        # Credit purposes with risk weights
        self.credit_purposes = {
            "car": {"weight": 0.30, "amount_range": (5000, 40000)},
            "furniture": {"weight": 0.15, "amount_range": (1000, 15000)},
            "radio/tv": {"weight": 0.10, "amount_range": (500, 5000)},
            "domestic": {"weight": 0.12, "amount_range": (2000, 20000)},
            "repairs": {"weight": 0.08, "amount_range": (1000, 25000)},
            "education": {"weight": 0.10, "amount_range": (2000, 30000)},
            "vacation": {"weight": 0.08, "amount_range": (1000, 10000)},
            "business": {"weight": 0.05, "amount_range": (10000, 100000)},
            "others": {"weight": 0.02, "amount_range": (1000, 50000)},
        }

        # Credit history patterns
        self.credit_history_types = {
            "no_credits": {"weight": 0.05, "risk_factor": 0.8},
            "all_paid": {"weight": 0.40, "risk_factor": 0.2},
            "existing_paid": {"weight": 0.35, "risk_factor": 0.3},
            "delayed_previously": {"weight": 0.15, "risk_factor": 0.7},
            "critical_other_existing": {"weight": 0.05, "risk_factor": 0.9},
        }

        # Housing types
        self.housing_types = {
            "rent": {"weight": 0.50, "stability_factor": 0.4},
            "own": {"weight": 0.40, "stability_factor": 0.8},
            "for_free": {"weight": 0.10, "stability_factor": 0.6},
        }

    def generate_dataset(
        self, n_samples: int = 1000, default_rate: float = 0.3
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate a complete synthetic credit dataset.

        Parameters:
        -----------
        n_samples : int
            Number of credit applications to generate
        default_rate : float
            Target proportion of defaults (bad credit risk)

        Returns:
        --------
        X : DataFrame
            Feature matrix with credit application data
        y : Series
            Target variable (0=good credit, 1=bad credit)
        """
        # Reset random seed for reproducible generation on each call
        self._reset_random_state()

        logger.info(f"Generating {n_samples} synthetic credit applications")

        # Generate base features
        data = {}

        # Demographics
        data["age"] = self._generate_ages(n_samples)
        data["gender"] = np.random.choice(["male", "female"], n_samples, p=[0.55, 0.45])

        # Employment information
        employment_info = self._generate_employment_data(n_samples)
        data.update(employment_info)

        # Housing and residence
        housing_info = self._generate_housing_data(n_samples)
        data.update(housing_info)

        # Credit request details
        credit_info = self._generate_credit_request_data(n_samples)
        data.update(credit_info)

        # Financial information
        financial_info = self._generate_financial_data(n_samples, data["credit_amount"])
        data.update(financial_info)

        # Credit history
        history_info = self._generate_credit_history_data(n_samples)
        data.update(history_info)

        # Create DataFrame
        X = pd.DataFrame(data)

        # Generate target variable based on risk factors
        y = self._generate_target_variable(X, default_rate)

        logger.info(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Default rate: {y.mean():.2%}")

        return X, y

    def _generate_ages(self, n_samples: int) -> np.ndarray:
        """Generate realistic age distribution."""
        ages = []
        for (min_age, max_age), weight in self.age_weights.items():
            count = int(n_samples * weight)
            ages.extend(np.random.randint(min_age, max_age + 1, count))

        # Fill remaining samples
        while len(ages) < n_samples:
            ages.append(np.random.randint(18, 76))

        return np.array(ages[:n_samples])

    def _generate_employment_data(self, n_samples: int) -> dict:
        """Generate employment-related features."""
        employment_data = {}

        # Employment duration
        durations = []
        employment_status = []

        for status, info in self.employment_patterns.items():
            count = int(n_samples * info["weight"])

            if status == "unemployed":
                durations.extend([0] * count)
                employment_status.extend(["unemployed"] * count)
            else:
                min_dur, max_dur = info["duration"]
                durations.extend(np.random.randint(min_dur, max_dur + 1, count))
                employment_status.extend([status] * count)

        # Fill remaining samples
        while len(durations) < n_samples:
            durations.append(np.random.randint(1, 120))
            employment_status.append("1_to_4_years")

        employment_data["employment_duration"] = np.array(durations[:n_samples])
        employment_data["employment_status"] = employment_status[:n_samples]

        # Job categories
        job_categories = ["unemployed", "unskilled", "skilled", "management"]
        job_weights = [0.05, 0.25, 0.55, 0.15]
        employment_data["job"] = np.random.choice(
            job_categories, n_samples, p=job_weights
        )

        return employment_data

    def _generate_housing_data(self, n_samples: int) -> dict:
        """Generate housing and residence information."""
        housing_data = {}

        # Housing type
        housing_types = list(self.housing_types.keys())
        housing_weights = [info["weight"] for info in self.housing_types.values()]
        housing_data["housing"] = np.random.choice(
            housing_types, n_samples, p=housing_weights
        )

        # Residence duration (correlated with age and housing type)
        residence_durations = []
        for i in range(n_samples):
            housing_type = housing_data["housing"][i]
            if housing_type == "own":
                # Homeowners tend to stay longer
                duration = np.random.exponential(48) + 12  # At least 1 year
            elif housing_type == "rent":
                # Renters move more frequently
                duration = np.random.exponential(24) + 6  # At least 6 months
            else:  # for_free
                # Living with family/friends - varies widely
                duration = np.random.exponential(36) + 3  # At least 3 months

            residence_durations.append(min(duration, 240))  # Cap at 20 years

        housing_data["residence_duration"] = np.array(residence_durations)

        return housing_data

    def _generate_credit_request_data(self, n_samples: int) -> dict:
        """Generate credit request details."""
        credit_data = {}

        # Credit purpose
        purposes = list(self.credit_purposes.keys())
        purpose_weights = [info["weight"] for info in self.credit_purposes.values()]
        credit_data["purpose"] = np.random.choice(
            purposes, n_samples, p=purpose_weights
        )

        # Credit amount (based on purpose)
        amounts = []
        for purpose in credit_data["purpose"]:
            min_amount, max_amount = self.credit_purposes[purpose]["amount_range"]
            # Log-normal distribution for realistic amount distribution
            log_mean = np.log((min_amount + max_amount) / 2)
            log_std = 0.5
            amount = np.random.lognormal(log_mean, log_std)
            amount = np.clip(amount, min_amount, max_amount)
            amounts.append(round(amount, -2))  # Round to nearest 100

        credit_data["credit_amount"] = np.array(amounts)

        # Credit duration (correlated with amount)
        durations = []
        for amount in credit_data["credit_amount"]:
            if amount < 5000:
                duration = np.random.randint(6, 25)  # 6-24 months
            elif amount < 20000:
                duration = np.random.randint(12, 49)  # 12-48 months
            else:
                duration = np.random.randint(24, 85)  # 24-84 months
            durations.append(duration)

        credit_data["duration"] = np.array(durations)

        return credit_data

    def _generate_financial_data(
        self, n_samples: int, credit_amounts: np.ndarray
    ) -> dict:
        """Generate financial information."""
        financial_data = {}

        # Monthly income (correlated with credit amount requested)
        incomes = []
        for amount in credit_amounts:
            # People typically request 3-20x their monthly income
            base_income = amount / np.random.uniform(3, 20)
            # Add some noise
            income = base_income * np.random.lognormal(0, 0.3)
            incomes.append(max(income, 500))  # Minimum income

        financial_data["monthly_income"] = np.array(incomes)

        # Existing credits
        financial_data["existing_credits"] = np.random.choice(
            [1, 2, 3, 4], n_samples, p=[0.60, 0.25, 0.12, 0.03]
        )

        # Savings account balance categories
        savings_categories = ["none", "low", "medium", "high"]
        savings_weights = [0.25, 0.40, 0.25, 0.10]
        financial_data["savings_status"] = np.random.choice(
            savings_categories, n_samples, p=savings_weights
        )

        # Checking account status
        checking_categories = ["none", "low", "medium"]
        checking_weights = [0.15, 0.70, 0.15]
        financial_data["checking_status"] = np.random.choice(
            checking_categories, n_samples, p=checking_weights
        )

        return financial_data

    def _generate_credit_history_data(self, n_samples: int) -> dict:
        """Generate credit history information."""
        history_data = {}

        # Credit history type
        history_types = list(self.credit_history_types.keys())
        history_weights = [
            info["weight"] for info in self.credit_history_types.values()
        ]
        history_data["credit_history"] = np.random.choice(
            history_types, n_samples, p=history_weights
        )

        # Number of people liable for maintenance
        history_data["num_dependents"] = np.random.choice(
            [1, 2, 3], n_samples, p=[0.70, 0.25, 0.05]
        )

        # Telephone (indicator of stability)
        history_data["telephone"] = np.random.choice(
            ["none", "yes"], n_samples, p=[0.20, 0.80]
        )

        # Foreign worker status
        history_data["foreign_worker"] = np.random.choice(
            ["yes", "no"], n_samples, p=[0.04, 0.96]
        )

        return history_data

    def _generate_target_variable(
        self, X: pd.DataFrame, target_default_rate: float
    ) -> pd.Series:
        """
        Generate target variable based on risk factors in the features.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        target_default_rate : float
            Desired proportion of defaults

        Returns:
        --------
        y : Series
            Binary target (0=good credit, 1=bad credit)
        """
        n_samples = len(X)
        risk_scores = np.zeros(n_samples)

        # Age risk (U-shaped: young and old = higher risk)
        age_risk = np.where(X["age"] < 25, 0.3, np.where(X["age"] > 65, 0.2, 0.0))
        risk_scores += age_risk

        # Employment duration risk
        employment_risk = np.where(
            X["employment_duration"] < 12,
            0.3,
            np.where(X["employment_duration"] < 48, 0.1, 0.0),
        )
        risk_scores += employment_risk

        # Credit history risk
        history_risk_map = {
            hist_type: info["risk_factor"]
            for hist_type, info in self.credit_history_types.items()
        }
        history_risk = X["credit_history"].map(history_risk_map)
        risk_scores += history_risk * 0.4

        # Debt-to-income risk
        debt_to_income = X["credit_amount"] / (X["monthly_income"] * 12)
        debt_risk = np.clip(debt_to_income - 0.3, 0, 1) * 0.3
        risk_scores += debt_risk

        # Purpose risk
        purpose_risk_map = {
            "business": 0.4,
            "vacation": 0.3,
            "others": 0.2,
            "domestic": 0.1,
            "repairs": 0.1,
            "education": 0.05,
            "car": 0.05,
            "furniture": 0.05,
            "radio/tv": 0.05,
        }
        purpose_risk = X["purpose"].map(purpose_risk_map)
        risk_scores += purpose_risk

        # Convert risk scores to probabilities using logistic function
        risk_probabilities = 1 / (1 + np.exp(-5 * (risk_scores - 0.5)))

        # Adjust to match target default rate
        current_rate = risk_probabilities.mean()
        adjustment = target_default_rate / current_rate
        adjusted_probabilities = np.clip(risk_probabilities * adjustment, 0, 1)

        # Generate binary outcomes
        y = np.random.binomial(1, adjusted_probabilities)

        return pd.Series(y, name="default_risk")

    def save_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        filename: str = "german_credit_synthetic.csv",
        include_target: bool = True,
    ) -> str:
        """
        Save the generated dataset to CSV file.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        filename : str
            Output filename
        include_target : bool
            Whether to include target variable in the saved file

        Returns:
        --------
        filepath : str
            Path to saved file
        """
        if include_target:
            data = X.copy()
            data["default_risk"] = y
        else:
            data = X

        filepath = f"demos/credit_risk/data/{filename}"
        data.to_csv(filepath, index=False)

        logger.info(f"Dataset saved to {filepath}")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")

        return filepath


def main():
    """Generate and save sample credit risk datasets."""

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize generator
    generator = CreditDataGenerator(random_seed=42)

    # Generate training dataset
    print("Generating training dataset...")
    X_train, y_train = generator.generate_dataset(n_samples=800, default_rate=0.30)
    train_path = generator.save_dataset(X_train, y_train, "credit_risk_train.csv")

    # Generate test dataset
    print("Generating test dataset...")
    X_test, y_test = generator.generate_dataset(n_samples=200, default_rate=0.30)
    test_path = generator.save_dataset(X_test, y_test, "credit_risk_test.csv")

    # Generate features-only dataset for prediction examples
    print("Generating prediction examples...")
    X_pred, _ = generator.generate_dataset(n_samples=50, default_rate=0.30)
    pred_path = generator.save_dataset(
        X_pred, None, "credit_risk_examples.csv", include_target=False
    )

    print(f"\nGenerated datasets:")
    print(f"Training: {train_path}")
    print(f"Test: {test_path}")
    print(f"Examples: {pred_path}")

    # Display sample statistics
    print(f"\nTraining set statistics:")
    print(f"Shape: {X_train.shape}")
    print(f"Default rate: {y_train.mean():.2%}")
    print(f"Average credit amount: ${X_train['credit_amount'].mean():,.0f}")
    print(f"Average age: {X_train['age'].mean():.1f} years")


if __name__ == "__main__":
    main()
