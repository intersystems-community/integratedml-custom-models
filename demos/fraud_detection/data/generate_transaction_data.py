"""
Realistic Transaction Data Generator for Fraud Detection Demo.

This module generates synthetic financial transaction data with realistic patterns
and sophisticated fraud scenarios for testing the ensemble fraud detection model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransactionPattern:
    """Configuration for different transaction patterns."""
    merchant_categories: List[str]
    amount_range: Tuple[float, float]
    time_preferences: List[int]  # Hours of day
    frequency_per_day: Tuple[int, int]  # Min, max transactions per day
    geographic_radius: float  # Miles from home location


class TransactionDataGenerator:
    """
    Generate realistic financial transaction data with fraud patterns.
    
    This generator creates synthetic transaction data that includes:
    - Realistic spending patterns for different customer segments
    - Geographic and temporal consistency for legitimate users
    - Sophisticated fraud patterns (account takeover, synthetic identity, etc.)
    - Real-time streaming simulation capabilities
    
    Key Features:
    - Multiple customer personas with distinct spending behaviors
    - Seasonal and temporal transaction patterns
    - Geographic constraints and anomalies
    - Various fraud attack vectors
    - Configurable fraud rates and complexity
    """
    
    def __init__(self,
                 n_customers: int = 10000,
                 fraud_rate: float = 0.02,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the transaction generator.
        
        Parameters:
        -----------
        n_customers : int, default=10000
            Number of unique customers to simulate
        fraud_rate : float, default=0.02
            Proportion of transactions that should be fraudulent
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.fraud_rate = fraud_rate
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        random.seed(random_state)

        # Backward-compatibility aliases expected by notebooks/tests
        # - num_customers: alias for n_customers
        # - num_merchants: accepted for compatibility (not strictly used)
        if 'num_customers' in kwargs and kwargs['num_customers'] is not None:
            try:
                self.n_customers = int(kwargs['num_customers'])
            except Exception:
                pass
        # Accept but do not require usage; keep for interface compatibility
        self._num_merchants_requested = kwargs.get('num_merchants', None)
        
        # Initialize data structures
        self.customers = {}
        self.merchant_data = {}
        self.transaction_patterns = {}
        self.fraud_patterns = {}
        
        # Generate base data
        self._initialize_merchants()
        self._initialize_customers()
        self._initialize_fraud_patterns()
        
    def _initialize_merchants(self) -> None:
        """Initialize merchant categories and characteristics."""
        self.merchant_categories = {
            'grocery': {
                'avg_amount': 75.0,
                'amount_std': 25.0,
                'min_amount': 5.0,
                'max_amount': 300.0,
                'frequency_weight': 3.0
            },
            'gas_station': {
                'avg_amount': 45.0,
                'amount_std': 15.0,
                'min_amount': 20.0,
                'max_amount': 100.0,
                'frequency_weight': 2.0
            },
            'restaurant': {
                'avg_amount': 35.0,
                'amount_std': 20.0,
                'min_amount': 8.0,
                'max_amount': 200.0,
                'frequency_weight': 2.5
            },
            'retail': {
                'avg_amount': 85.0,
                'amount_std': 60.0,
                'min_amount': 10.0,
                'max_amount': 500.0,
                'frequency_weight': 1.5
            },
            'online': {
                'avg_amount': 65.0,
                'amount_std': 45.0,
                'min_amount': 5.0,
                'max_amount': 1000.0,
                'frequency_weight': 2.0
            },
            'entertainment': {
                'avg_amount': 28.0,
                'amount_std': 15.0,
                'min_amount': 10.0,
                'max_amount': 100.0,
                'frequency_weight': 1.0
            },
            'travel': {
                'avg_amount': 450.0,
                'amount_std': 300.0,
                'min_amount': 50.0,
                'max_amount': 2000.0,
                'frequency_weight': 0.2
            },
            'luxury': {
                'avg_amount': 1200.0,
                'amount_std': 800.0,
                'min_amount': 200.0,
                'max_amount': 5000.0,
                'frequency_weight': 0.1
            }
        }
        
    def _initialize_customers(self) -> None:
        """Initialize customer profiles with realistic characteristics."""
        customer_types = ['conservative', 'moderate', 'active', 'luxury', 'young_adult']
        
        for customer_id in range(self.n_customers):
            customer_type = np.random.choice(customer_types, p=[0.3, 0.4, 0.2, 0.05, 0.05])
            
            # Base customer attributes
            self.customers[customer_id] = {
                'customer_id': customer_id,
                'customer_type': customer_type,
                'home_lat': np.random.uniform(25.0, 49.0),  # Continental US
                'home_lon': np.random.uniform(-125.0, -66.0),
                'income_level': self._get_income_level(customer_type),
                'credit_limit': self._get_credit_limit(customer_type),
                'account_age_days': np.random.randint(30, 3650),  # 1 month to 10 years
                'preferred_categories': self._get_preferred_categories(customer_type),
                'daily_transaction_limit': self._get_daily_limit(customer_type),
                'spending_velocity': self._get_spending_velocity(customer_type)
            }
            
    def _get_income_level(self, customer_type: str) -> str:
        """Get income level based on customer type."""
        income_mapping = {
            'conservative': np.random.choice(['low', 'medium'], p=[0.6, 0.4]),
            'moderate': np.random.choice(['medium', 'high'], p=[0.7, 0.3]),
            'active': np.random.choice(['medium', 'high'], p=[0.5, 0.5]),
            'luxury': np.random.choice(['high', 'very_high'], p=[0.3, 0.7]),
            'young_adult': np.random.choice(['low', 'medium'], p=[0.8, 0.2])
        }
        return income_mapping[customer_type]
        
    def _get_credit_limit(self, customer_type: str) -> float:
        """Get credit limit based on customer type."""
        limits = {
            'conservative': np.random.uniform(1000, 5000),
            'moderate': np.random.uniform(3000, 15000),
            'active': np.random.uniform(10000, 30000),
            'luxury': np.random.uniform(25000, 100000),
            'young_adult': np.random.uniform(500, 3000)
        }
        return limits[customer_type]
        
    def _get_preferred_categories(self, customer_type: str) -> List[str]:
        """Get preferred merchant categories for customer type."""
        preferences = {
            'conservative': ['grocery', 'gas_station', 'restaurant'],
            'moderate': ['grocery', 'gas_station', 'restaurant', 'retail', 'online'],
            'active': ['restaurant', 'retail', 'online', 'entertainment', 'travel'],
            'luxury': ['restaurant', 'retail', 'travel', 'luxury', 'entertainment'],
            'young_adult': ['restaurant', 'online', 'entertainment', 'gas_station']
        }
        return preferences[customer_type]
        
    def _get_daily_limit(self, customer_type: str) -> int:
        """Get daily transaction limit for customer type."""
        limits = {
            'conservative': np.random.randint(2, 6),
            'moderate': np.random.randint(3, 8),
            'active': np.random.randint(5, 12),
            'luxury': np.random.randint(3, 15),
            'young_adult': np.random.randint(4, 10)
        }
        return limits[customer_type]
        
    def _get_spending_velocity(self, customer_type: str) -> float:
        """Get spending velocity multiplier for customer type."""
        velocities = {
            'conservative': np.random.uniform(0.3, 0.7),
            'moderate': np.random.uniform(0.7, 1.2),
            'active': np.random.uniform(1.0, 1.8),
            'luxury': np.random.uniform(1.5, 3.0),
            'young_adult': np.random.uniform(0.5, 1.3)
        }
        return velocities[customer_type]
        
    def _initialize_fraud_patterns(self) -> None:
        """Initialize different types of fraud attack patterns."""
        self.fraud_patterns = {
            'account_takeover': {
                'description': 'Legitimate account compromised by fraudster',
                'characteristics': {
                    'location_change': True,
                    'amount_escalation': True,
                    'category_change': True,
                    'time_pattern_change': True,
                    'velocity_increase': True
                }
            },
            'card_testing': {
                'description': 'Small transactions to test stolen card validity',
                'characteristics': {
                    'small_amounts': True,
                    'multiple_merchants': True,
                    'rapid_succession': True,
                    'online_heavy': True
                }
            },
            'synthetic_identity': {
                'description': 'Fake identity created with real/fake information',
                'characteristics': {
                    'new_account': True,
                    'unusual_patterns': True,
                    'high_value_early': True,
                    'limited_history': True
                }
            },
            'velocity_fraud': {
                'description': 'Rapid sequence of transactions',
                'characteristics': {
                    'burst_activity': True,
                    'multiple_locations': True,
                    'amount_escalation': True,
                    'short_timeframe': True
                }
            },
            'manual_entry_fraud': {
                'description': 'Card-not-present fraud with manually entered details',
                'characteristics': {
                    'online_only': True,
                    'no_chip_pin': True,
                    'suspicious_amounts': True,
                    'merchant_category_specific': True
                }
            }
        }
        
    def generate_transactions(self, 
                            n_transactions: int = 100000,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate realistic transaction dataset.
        
        Parameters:
        -----------
        n_transactions : int, default=100000
            Number of transactions to generate
        start_date : datetime, optional
            Start date for transaction range
        end_date : datetime, optional
            End date for transaction range
            
        Returns:
        --------
        transactions : DataFrame
            Generated transaction data with fraud labels
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()
            
        transactions = []
        n_fraud_transactions = int(n_transactions * self.fraud_rate)
        n_legitimate_transactions = n_transactions - n_fraud_transactions
        
        # Generate legitimate transactions
        logger.info(f"Generating {n_legitimate_transactions} legitimate transactions...")
        legitimate_transactions = self._generate_legitimate_transactions(
            n_legitimate_transactions, start_date, end_date
        )
        transactions.extend(legitimate_transactions)
        
        # Generate fraudulent transactions
        logger.info(f"Generating {n_fraud_transactions} fraudulent transactions...")
        fraudulent_transactions = self._generate_fraudulent_transactions(
            n_fraud_transactions, start_date, end_date
        )
        transactions.extend(fraudulent_transactions)
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Add transaction IDs
        df['transaction_id'] = range(len(df))
        
        # Add derived features for fraud detection
        df = self._add_derived_features(df)
        
        logger.info(f"Generated {len(df)} total transactions ({df['is_fraud'].sum()} fraudulent)")
        
        return df
        
    def _generate_legitimate_transactions(self, 
                                        n_transactions: int,
                                        start_date: datetime,
                                        end_date: datetime) -> List[Dict]:
        """Generate realistic legitimate transactions."""
        transactions = []
        
        for _ in range(n_transactions):
            # Select random customer
            customer_id = np.random.randint(0, self.n_customers)
            customer = self.customers[customer_id]
            
            # Generate transaction timestamp
            timestamp = self._generate_realistic_timestamp(customer, start_date, end_date)
            
            # Select merchant category based on customer preferences
            category = np.random.choice(
                customer['preferred_categories'],
                p=self._get_category_probabilities(customer['preferred_categories'])
            )
            
            # Generate transaction amount
            amount = self._generate_realistic_amount(customer, category)
            
            # Generate location (near customer's home for legitimate transactions)
            lat, lon = self._generate_location_near_home(customer)
            
            # Create transaction record
            transaction = {
                'customer_id': customer_id,
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'merchant_category': category,
                'merchant_id': self._get_random_merchant_id(category),
                'latitude': lat,
                'longitude': lon,
                'is_chip_transaction': np.random.choice([True, False], p=[0.8, 0.2]),
                'is_pin_transaction': np.random.choice([True, False], p=[0.6, 0.4]),
                'is_online_transaction': category == 'online',
                'is_fraud': 0,
                'fraud_type': None
            }
            
            transactions.append(transaction)
            
        return transactions
        
    def _generate_fraudulent_transactions(self,
                                        n_transactions: int,
                                        start_date: datetime,
                                        end_date: datetime) -> List[Dict]:
        """Generate fraudulent transactions with realistic attack patterns."""
        transactions = []
        fraud_types = list(self.fraud_patterns.keys())
        
        for _ in range(n_transactions):
            # Select fraud type
            fraud_type = np.random.choice(fraud_types, p=[0.4, 0.2, 0.15, 0.15, 0.1])
            
            # Select target customer (victim)
            customer_id = np.random.randint(0, self.n_customers)
            customer = self.customers[customer_id]
            
            # Generate fraudulent transaction based on pattern
            transaction = self._generate_fraud_transaction(
                customer, fraud_type, start_date, end_date
            )
            
            transactions.append(transaction)
            
        return transactions
        
    def _generate_fraud_transaction(self,
                                  customer: Dict,
                                  fraud_type: str,
                                  start_date: datetime,
                                  end_date: datetime) -> Dict:
        """Generate a single fraudulent transaction based on fraud pattern."""
        pattern = self.fraud_patterns[fraud_type]
        
        if fraud_type == 'account_takeover':
            return self._generate_account_takeover_transaction(customer, start_date, end_date)
        elif fraud_type == 'card_testing':
            return self._generate_card_testing_transaction(customer, start_date, end_date)
        elif fraud_type == 'synthetic_identity':
            return self._generate_synthetic_identity_transaction(customer, start_date, end_date)
        elif fraud_type == 'velocity_fraud':
            return self._generate_velocity_fraud_transaction(customer, start_date, end_date)
        elif fraud_type == 'manual_entry_fraud':
            return self._generate_manual_entry_fraud_transaction(customer, start_date, end_date)
        else:
            # Default to account takeover
            return self._generate_account_takeover_transaction(customer, start_date, end_date)
            
    def _generate_account_takeover_transaction(self,
                                             customer: Dict,
                                             start_date: datetime,
                                             end_date: datetime) -> Dict:
        """Generate account takeover fraud transaction."""
        # Fraudulent location (far from home)
        lat = np.random.uniform(25.0, 49.0)
        lon = np.random.uniform(-125.0, -66.0)
        
        # Unusual merchant category
        unusual_categories = ['luxury', 'travel', 'online']
        category = np.random.choice(unusual_categories)
        
        # Higher than normal amount
        base_amount = self.merchant_categories[category]['avg_amount']
        amount = base_amount * np.random.uniform(2.0, 5.0)
        
        # Unusual time
        timestamp = self._generate_unusual_timestamp(start_date, end_date)
        
        return {
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'merchant_id': self._get_random_merchant_id(category),
            'latitude': lat,
            'longitude': lon,
            'is_chip_transaction': False,  # Often card-not-present
            'is_pin_transaction': False,
            'is_online_transaction': np.random.choice([True, False], p=[0.7, 0.3]),
            'is_fraud': 1,
            'fraud_type': 'account_takeover'
        }
        
    def _generate_card_testing_transaction(self,
                                         customer: Dict,
                                         start_date: datetime,
                                         end_date: datetime) -> Dict:
        """Generate card testing fraud transaction."""
        # Small amounts for testing
        amount = np.random.uniform(1.0, 10.0)
        
        # Online transactions
        category = 'online'
        
        # Random location
        lat = np.random.uniform(25.0, 49.0)
        lon = np.random.uniform(-125.0, -66.0)
        
        # Random timestamp
        timestamp = start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return {
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'merchant_id': self._get_random_merchant_id(category),
            'latitude': lat,
            'longitude': lon,
            'is_chip_transaction': False,
            'is_pin_transaction': False,
            'is_online_transaction': True,
            'is_fraud': 1,
            'fraud_type': 'card_testing'
        }
        
    def _generate_synthetic_identity_transaction(self,
                                               customer: Dict,
                                               start_date: datetime,
                                               end_date: datetime) -> Dict:
        """Generate synthetic identity fraud transaction."""
        # High-value transaction early in account lifecycle
        amount = np.random.uniform(500.0, 2000.0)
        
        # Luxury or travel category
        category = np.random.choice(['luxury', 'travel', 'retail'])
        
        # Random location
        lat = np.random.uniform(25.0, 49.0)
        lon = np.random.uniform(-125.0, -66.0)
        
        # Recent timestamp (new account)
        recent_start = end_date - timedelta(days=30)
        timestamp = recent_start + timedelta(
            seconds=np.random.randint(0, int((end_date - recent_start).total_seconds()))
        )
        
        return {
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'merchant_id': self._get_random_merchant_id(category),
            'latitude': lat,
            'longitude': lon,
            'is_chip_transaction': np.random.choice([True, False], p=[0.3, 0.7]),
            'is_pin_transaction': np.random.choice([True, False], p=[0.2, 0.8]),
            'is_online_transaction': np.random.choice([True, False], p=[0.6, 0.4]),
            'is_fraud': 1,
            'fraud_type': 'synthetic_identity'
        }
        
    def _generate_velocity_fraud_transaction(self,
                                           customer: Dict,
                                           start_date: datetime,
                                           end_date: datetime) -> Dict:
        """Generate velocity fraud transaction."""
        # High amount
        amount = np.random.uniform(200.0, 1000.0)
        
        # Random category
        category = np.random.choice(list(self.merchant_categories.keys()))
        
        # Multiple locations possible
        lat = np.random.uniform(25.0, 49.0)
        lon = np.random.uniform(-125.0, -66.0)
        
        # Clustered timestamps
        timestamp = start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return {
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'merchant_id': self._get_random_merchant_id(category),
            'latitude': lat,
            'longitude': lon,
            'is_chip_transaction': np.random.choice([True, False], p=[0.4, 0.6]),
            'is_pin_transaction': np.random.choice([True, False], p=[0.3, 0.7]),
            'is_online_transaction': category == 'online',
            'is_fraud': 1,
            'fraud_type': 'velocity_fraud'
        }
        
    def _generate_manual_entry_fraud_transaction(self,
                                               customer: Dict,
                                               start_date: datetime,
                                               end_date: datetime) -> Dict:
        """Generate manual entry fraud transaction."""
        # Medium to high amounts
        amount = np.random.uniform(100.0, 800.0)
        
        # Online transactions
        category = 'online'
        
        # Random location
        lat = np.random.uniform(25.0, 49.0)
        lon = np.random.uniform(-125.0, -66.0)
        
        # Random timestamp
        timestamp = start_date + timedelta(
            seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return {
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'merchant_id': self._get_random_merchant_id(category),
            'latitude': lat,
            'longitude': lon,
            'is_chip_transaction': False,
            'is_pin_transaction': False,
            'is_online_transaction': True,
            'is_fraud': 1,
            'fraud_type': 'manual_entry_fraud'
        }
        
    def _generate_realistic_timestamp(self,
                                    customer: Dict,
                                    start_date: datetime,
                                    end_date: datetime) -> datetime:
        """Generate realistic timestamp based on customer patterns."""
        # Choose random day
        total_days = (end_date - start_date).days
        random_day = start_date + timedelta(days=np.random.randint(0, total_days))
        
        # Choose realistic hour based on customer type
        if customer['customer_type'] == 'young_adult':
            # More likely to shop in evenings and weekends
            hour = np.random.choice(range(24), p=self._get_young_adult_time_probs())
        elif customer['customer_type'] == 'conservative':
            # More likely to shop during business hours
            hour = np.random.choice(range(24), p=self._get_conservative_time_probs())
        else:
            # General population
            hour = np.random.choice(range(24), p=self._get_general_time_probs())
            
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return random_day.replace(hour=hour, minute=minute, second=second)
        
    def _generate_unusual_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate unusual timestamp for fraud (e.g., 3 AM)."""
        total_seconds = int((end_date - start_date).total_seconds())
        random_timestamp = start_date + timedelta(seconds=np.random.randint(0, total_seconds))
        
        # Set to unusual hour (late night/early morning)
        unusual_hour = np.random.choice([1, 2, 3, 4, 5, 23], p=[0.2, 0.3, 0.3, 0.1, 0.05, 0.05])
        
        return random_timestamp.replace(
            hour=unusual_hour,
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
    def _get_young_adult_time_probs(self) -> np.ndarray:
        """Get time preferences for young adults."""
        probs = np.ones(24) * 0.02  # Base probability
        # Higher probability for evenings and late night
        probs[18:23] = 0.08
        probs[12:17] = 0.06
        probs[8:12] = 0.04
        return probs / probs.sum()
        
    def _get_conservative_time_probs(self) -> np.ndarray:
        """Get time preferences for conservative customers."""
        probs = np.ones(24) * 0.01  # Base probability
        # Higher probability for business hours and early evening
        probs[9:17] = 0.08
        probs[17:21] = 0.06
        probs[7:9] = 0.03
        return probs / probs.sum()
        
    def _get_general_time_probs(self) -> np.ndarray:
        """Get general time preferences."""
        probs = np.ones(24) * 0.02  # Base probability
        # Higher probability for typical shopping hours
        probs[10:20] = 0.06
        probs[20:22] = 0.04
        probs[8:10] = 0.03
        return probs / probs.sum()
        
    def _generate_realistic_amount(self, customer: Dict, category: str) -> float:
        """Generate realistic transaction amount."""
        merchant_info = self.merchant_categories[category]
        
        # Base amount from merchant category
        base_amount = np.random.normal(
            merchant_info['avg_amount'],
            merchant_info['amount_std']
        )
        
        # Apply customer spending velocity
        amount = base_amount * customer['spending_velocity']
        
        # Apply constraints
        amount = max(merchant_info['min_amount'], amount)
        amount = min(merchant_info['max_amount'], amount)
        
        return amount
        
    def _generate_location_near_home(self, customer: Dict, max_distance: float = 50.0) -> Tuple[float, float]:
        """Generate transaction location near customer's home."""
        # Generate location within reasonable distance of home
        # Using simple rectangular approximation for speed
        lat_offset = np.random.uniform(-0.5, 0.5)  # Roughly 50 miles
        lon_offset = np.random.uniform(-0.5, 0.5)
        
        lat = customer['home_lat'] + lat_offset
        lon = customer['home_lon'] + lon_offset
        
        # Keep within US bounds
        lat = np.clip(lat, 25.0, 49.0)
        lon = np.clip(lon, -125.0, -66.0)
        
        return lat, lon
        
    def _get_category_probabilities(self, preferred_categories: List[str]) -> np.ndarray:
        """Get probability distribution for merchant categories."""
        n_categories = len(preferred_categories)
        # Slightly prefer first categories in the list
        probs = np.array([1.0 / i for i in range(1, n_categories + 1)])
        return probs / probs.sum()
        
    def _get_random_merchant_id(self, category: str) -> str:
        """Generate random merchant ID for category."""
        return f"{category}_{np.random.randint(1000, 9999)}"
        
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for fraud detection."""
        # Sort by customer and timestamp
        df = df.sort_values(['customer_id', 'timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        
        # Customer-based features (calculated per customer)
        customer_features = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'count'],
            'merchant_category': lambda x: len(set(x)),
            'timestamp': ['min', 'max']
        }).round(2)
        
        customer_features.columns = [
            'customer_avg_amount', 'customer_amount_std', 'customer_transaction_count',
            'customer_unique_categories', 'customer_first_transaction', 'customer_last_transaction'
        ]
        
        # Merge customer features
        df = df.merge(customer_features, left_on='customer_id', right_index=True)
        
        # Amount deviation from customer norm
        df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_amount_std'] + 1e-8)
        
        # Transaction velocity features (transactions per day)
        df['days_since_first'] = (df['timestamp'] - df['customer_first_transaction']).dt.days + 1
        df['transaction_velocity'] = df['customer_transaction_count'] / df['days_since_first']
        
        return df
        
    def generate_streaming_data(self, 
                              transactions_per_minute: int = 100,
                              duration_minutes: int = 60) -> pd.DataFrame:
        """
        Generate streaming transaction data for real-time testing.
        
        Parameters:
        -----------
        transactions_per_minute : int, default=100
            Rate of transaction generation
        duration_minutes : int, default=60
            Duration of simulation in minutes
            
        Returns:
        --------
        streaming_data : DataFrame
            Transaction data with precise timestamps for streaming
        """
        total_transactions = transactions_per_minute * duration_minutes
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Generate transactions
        df = self.generate_transactions(
            n_transactions=total_transactions,
            start_date=start_time,
            end_date=end_time
        )
        
        # Ensure even distribution across time
        time_intervals = pd.date_range(start_time, end_time, periods=total_transactions)
        df['timestamp'] = time_intervals
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df


def create_sample_dataset(output_path: str = None, 
                         n_transactions: int = 50000,
                         fraud_rate: float = 0.02) -> pd.DataFrame:
    """
    Create a sample transaction dataset for demo purposes.
    
    Parameters:
    -----------
    output_path : str, optional
        Path to save the dataset CSV file
    n_transactions : int, default=50000
        Number of transactions to generate
    fraud_rate : float, default=0.02
        Proportion of fraudulent transactions
        
    Returns:
    --------
    dataset : DataFrame
        Generated transaction dataset
    """
    logger.info(f"Creating sample dataset with {n_transactions} transactions...")
    
    generator = RealisticTransactionGenerator(
        n_customers=5000,
        fraud_rate=fraud_rate,
        random_state=42
    )
    
    dataset = generator.generate_transactions(n_transactions)
    
    if output_path:
        dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    
    return dataset


if __name__ == "__main__":
    # Generate sample dataset for testing
    sample_data = create_sample_dataset(
        output_path="demos/fraud_detection/data/sample_transactions.csv",
        n_transactions=10000,
        fraud_rate=0.03
    )
    
    print(f"Generated {len(sample_data)} transactions")
    print(f"Fraud rate: {sample_data['is_fraud'].mean():.3f}")
    print("\nSample transactions:")
    print(sample_data.head())
    
    print("\nFraud type distribution:")
    fraud_data = sample_data[sample_data['is_fraud'] == 1]
    if len(fraud_data) > 0:
        print(fraud_data['fraud_type'].value_counts())