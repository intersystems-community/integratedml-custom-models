"""
Data Preprocessing Utilities for Credit Risk Assessment

This module provides comprehensive data preprocessing and feature engineering
utilities for the credit risk assessment demo, including validation, cleaning,
and transformation functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class CreditDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for credit risk data.
    
    Handles data validation, cleaning, encoding, feature engineering,
    and preparation for machine learning models.
    """
    
    def __init__(self,
                 handle_missing: bool = True,
                 remove_outliers: bool = False,
                 normalize_features: bool = False,
                 outlier_method: str = 'iqr',
                 missing_strategy: str = 'median'):
        """Initialize the preprocessor with configurable settings."""
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.normalize_features = normalize_features
        self.outlier_method = outlier_method
        self.missing_strategy = missing_strategy
        
        self.label_encoders = {}
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.is_fitted = False
        self._preprocessing_metadata = {}
        
        # Define expected feature types and ranges
        self._setup_feature_specifications()
    
    def _setup_feature_specifications(self):
        """Define expected feature types and validation rules."""
        
        self.feature_specs = {
            # Demographic features
            'age': {'type': 'numeric', 'min': 18, 'max': 100},
            'gender': {'type': 'categorical', 'values': ['male', 'female']},
            
            # Employment features
            'employment_duration': {'type': 'numeric', 'min': 0, 'max': 600},
            'employment_status': {'type': 'categorical'},
            'job': {'type': 'categorical'},
            
            # Housing features
            'housing': {'type': 'categorical', 'values': ['rent', 'own', 'for_free']},
            'residence_duration': {'type': 'numeric', 'min': 0, 'max': 600},
            
            # Credit request features
            'credit_amount': {'type': 'numeric', 'min': 100, 'max': 1000000},
            'duration': {'type': 'numeric', 'min': 6, 'max': 120},
            'purpose': {'type': 'categorical'},
            
            # Financial features
            'monthly_income': {'type': 'numeric', 'min': 0, 'max': 1000000},
            'existing_credits': {'type': 'numeric', 'min': 1, 'max': 10},
            'savings_status': {'type': 'categorical'},
            'checking_status': {'type': 'categorical'},
            
            # Credit history features
            'credit_history': {'type': 'categorical'},
            'num_dependents': {'type': 'numeric', 'min': 0, 'max': 10},
            'telephone': {'type': 'categorical', 'values': ['none', 'yes']},
            'foreign_worker': {'type': 'categorical', 'values': ['yes', 'no']}
        }
    
    def validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, List[str]]:
        """
        Validate input data against expected specifications.
        
        Parameters:
        -----------
        X : DataFrame
            Input data to validate
        y : Series, optional
            Target variable to validate
            
        Returns:
        --------
        validation_results : dict
            Dictionary with validation issues categorized by type
        """
        issues = {
            'missing_features': [],
            'invalid_types': [],
            'out_of_range': [],
            'invalid_categories': [],
            'warnings': [],
            'data_types': {},
            'missing_values': {'total_missing': 0, 'by_column': {}},
            'feature_ranges': {},
            'target_distribution': {},
            'has_critical_issues': False
        }
        
        # Check for missing expected features
        expected_features = set(self.feature_specs.keys())
        actual_features = set(X.columns)
        missing_features = expected_features - actual_features
        if missing_features:
            issues['missing_features'] = list(missing_features)
        
        # Validate existing features
        for feature in actual_features.intersection(expected_features):
            spec = self.feature_specs[feature]
            
            # Check numeric features
            if spec['type'] == 'numeric':
                if not pd.api.types.is_numeric_dtype(X[feature]):
                    issues['invalid_types'].append(f"{feature}: expected numeric, got {X[feature].dtype}")
                else:
                    # Check range
                    if 'min' in spec and X[feature].min() < spec['min']:
                        issues['out_of_range'].append(f"{feature}: values below minimum {spec['min']}")
                    if 'max' in spec and X[feature].max() > spec['max']:
                        issues['out_of_range'].append(f"{feature}: values above maximum {spec['max']}")
            
            # Check categorical features
            elif spec['type'] == 'categorical':
                if 'values' in spec:
                    invalid_values = set(X[feature].dropna().unique()) - set(spec['values'])
                    if invalid_values:
                        issues['invalid_categories'].append(
                            f"{feature}: unexpected values {invalid_values}"
                        )
            
            # Check for missing values
            missing_count = X[feature].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(X) * 100
                if missing_pct > 50:
                    issues['warnings'].append(f"{feature}: {missing_pct:.1f}% missing values")
        
        # Calculate missing values statistics
        total_missing = X.isnull().sum().sum()
        issues['missing_values']['total_missing'] = total_missing
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                issues['missing_values']['by_column'][col] = missing_count
        
        # Add data types information
        for col in X.columns:
            issues['data_types'][col] = str(X[col].dtype)
            
            # Add feature ranges for numeric columns
            if pd.api.types.is_numeric_dtype(X[col]):
                issues['feature_ranges'][col] = {
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std())
                }
        
        # Add target distribution if y is provided
        if y is not None:
            issues['target_distribution'] = {
                'unique_values': y.nunique(),
                'value_counts': y.value_counts().to_dict(),
                'missing_targets': y.isnull().sum()
            }
        
        # Compile problematic features
        problematic_features = []
        problematic_features.extend(issues['missing_features'])
        problematic_features.extend([feat.split(':')[0] for feat in issues['invalid_types']])
        problematic_features.extend([feat.split(':')[0] for feat in issues['out_of_range']])
        problematic_features.extend([feat.split(':')[0] for feat in issues['invalid_categories']])
        issues['problematic_features'] = list(set(problematic_features))  # Remove duplicates
        
        # Determine if there are critical issues
        critical_issues = (
            len(issues['missing_features']) > 0 or
            len(issues['invalid_types']) > 0 or
            len(issues['out_of_range']) > 0 or
            total_missing > len(X) * 0.5  # More than 50% missing data
        )
        issues['has_critical_issues'] = critical_issues
        
        logger.info(f"Data validation completed. Found missing values: {total_missing}")
        logger.info(f"Data types: {issues['data_types']}")
        logger.info(f"Critical issues detected: {critical_issues}")
        return issues
    
    def _detect_outliers_zscore(self, X, threshold=3):
        """Detect outliers using Z-score method"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        outliers = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(X[col].fillna(X[col].median())))
            outliers[col] = np.where(z_scores > threshold)[0].tolist()
        return outliers
    
    @classmethod
    def load_preprocessor(cls, file_path):
        """Load a saved preprocessor from file"""
        import joblib
        try:
            return joblib.load(file_path)
        except Exception as e:
            logger.error(f"Failed to load preprocessor from {file_path}: {e}")
            raise
    
    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize input data.
        
        Parameters:
        -----------
        X : DataFrame
            Raw input data
            
        Returns:
        --------
        X_clean : DataFrame
            Cleaned data
        """
        X_clean = X.copy()
        
        # Remove duplicate rows first
        if len(X_clean) > 1:
            initial_rows = len(X_clean)
            X_clean = X_clean.drop_duplicates().reset_index(drop=True)
            removed_duplicates = initial_rows - len(X_clean)
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Standardize column names
        X_clean.columns = X_clean.columns.str.lower().str.replace(' ', '_')
        
        # Handle specific data cleaning rules
        
        # Age: clip to reasonable range
        if 'age' in X_clean.columns:
            X_clean['age'] = np.clip(X_clean['age'], 18, 100)
        
        # Credit amount: ensure positive
        if 'credit_amount' in X_clean.columns:
            X_clean['credit_amount'] = np.maximum(X_clean['credit_amount'], 100)
        
        # Duration: clip to reasonable range
        if 'duration' in X_clean.columns:
            X_clean['duration'] = np.clip(X_clean['duration'], 6, 120)
        
        # Employment duration: ensure non-negative
        if 'employment_duration' in X_clean.columns:
            X_clean['employment_duration'] = np.maximum(X_clean['employment_duration'], 0)
        
        # Monthly income: handle missing and ensure positive
        if 'monthly_income' in X_clean.columns:
            # Fill missing income based on credit amount
            if 'credit_amount' in X_clean.columns:
                missing_income = X_clean['monthly_income'].isnull()
                if missing_income.any():
                    estimated_income = X_clean.loc[missing_income, 'credit_amount'] / 10
                    X_clean.loc[missing_income, 'monthly_income'] = estimated_income
            
            X_clean['monthly_income'] = np.maximum(X_clean['monthly_income'], 500)
        
        # Standardize categorical values
        categorical_mappings = {
            'gender': {'m': 'male', 'f': 'female', 'male': 'male', 'female': 'female'},
            'housing': {'rent': 'rent', 'own': 'own', 'free': 'for_free', 'for_free': 'for_free'},
            'telephone': {'no': 'none', 'none': 'none', 'yes': 'yes'},
            'foreign_worker': {'yes': 'yes', 'no': 'no'}
        }
        
        for feature, mapping in categorical_mappings.items():
            if feature in X_clean.columns:
                X_clean[feature] = X_clean[feature].astype(str).str.lower().map(mapping)
                X_clean[feature] = X_clean[feature].fillna('unknown')
        
        logger.info(f"Data cleaning completed. Shape: {X_clean.shape}")
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        y : Series, optional
            Target variable (not used in preprocessing)
            
        Returns:
        --------
        X_transformed : DataFrame
            Transformed features
        y_transformed : Series, optional
            Target variable (aligned with X after duplicate removal)
        """
        # Clean data first and handle y alignment
        if y is not None:
            # Combine X and y to ensure duplicate removal is synchronized
            original_indices = X.index
            X_clean = self.clean_data(X)
            
            # Find which rows were removed during cleaning
            remaining_indices = X_clean.index
            y_clean = y.loc[remaining_indices] if len(remaining_indices) < len(original_indices) else y
        else:
            X_clean = self.clean_data(X)
            y_clean = None
        
        # Identify feature types
        self._feature_types = self._identify_feature_types(X_clean)
        
        # Apply transformations step by step for fitting
        X_transformed = X_clean.copy()
        
        # Step 1: Fit and apply imputation
        self.numeric_imputer = SimpleImputer(strategy=self.missing_strategy)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        if self.numerical_features:
            X_transformed[self.numerical_features] = self.numeric_imputer.fit_transform(
                X_transformed[self.numerical_features]
            )
        
        if self.categorical_features:
            X_transformed[self.categorical_features] = self.categorical_imputer.fit_transform(
                X_transformed[self.categorical_features]
            )
        
        # Step 2: Fit and apply encoding
        for feature in self.categorical_features:
            le = LabelEncoder()
            X_transformed[feature] = le.fit_transform(X_transformed[feature].astype(str))
            self.label_encoders[feature] = le
        
        # Step 3: Fit and apply scaling (on imputed and encoded data)
        if self.normalize_features and self.numerical_features:
            self.scaler = StandardScaler()
            X_transformed[self.numerical_features] = self.scaler.fit_transform(
                X_transformed[self.numerical_features]
            )
        
        # Store preprocessing metadata
        self._preprocessing_metadata = {
            'original_features': list(X.columns),
            'processed_features': list(X_transformed.columns),
            'transformed_features': list(X_transformed.columns),
            'feature_types': self._feature_types,
            'original_shape': X.shape,
            'transformed_shape': X_transformed.shape,
            'normalization_enabled': self.normalize_features
        }
        
        self.is_fitted = True
        logger.info("Preprocessor fitted and data transformed")
        return X_transformed, y_clean
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CreditDataPreprocessor':
        """
        Fit the preprocessor without transforming data.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
        y : Series, optional
            Target variable (not used in preprocessing)
            
        Returns:
        --------
        self : CreditDataPreprocessor
            Fitted preprocessor
        """
        # Clean data first
        X_clean = self.clean_data(X)
        
        # Identify feature types
        self._feature_types = self._identify_feature_types(X_clean)
        
        # Fit transformations without applying them
        self._fit_transformations(X_clean)
        
        # Store preprocessing metadata (same as in fit_transform)
        self._preprocessing_metadata = {
            'original_features': list(X.columns),
            'processed_features': list(X_clean.columns),
            'transformed_features': list(X_clean.columns),
            'feature_types': self._feature_types,
            'original_shape': X.shape,
            'transformed_shape': X_clean.shape,
            'normalization_enabled': self.normalize_features
        }
        
        self.is_fitted = True
        logger.info("Preprocessor fitted")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        X : DataFrame
            Input features
            
        Returns:
        --------
        X_transformed : DataFrame
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Clean data
        X_clean = self.clean_data(X)
        
        # Transform
        X_transformed = self._apply_transformations(X_clean, fit=False)
        
        return X_transformed
    
    def _fit_transformations(self, X: pd.DataFrame) -> None:
        """Fit preprocessing transformations without applying them."""
        # Fit imputers
        self.numeric_imputer = SimpleImputer(strategy=self.missing_strategy)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        if self.numerical_features:
            self.numeric_imputer.fit(X[self.numerical_features])
        
        if self.categorical_features:
            self.categorical_imputer.fit(X[self.categorical_features])
        
        # Fit label encoders
        for feature in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[feature].astype(str))
            self.label_encoders[feature] = le
        
        # Fit scaler if needed
        if self.normalize_features and self.numerical_features:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numerical_features])
    
    def _identify_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify numerical, categorical, and binary features."""
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        
        for col in X.columns:
            unique_values = X[col].nunique()
            
            # Check for binary features (2 unique values, regardless of type)
            if unique_values == 2:
                self.binary_features.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.debug(f"Identified {len(self.numerical_features)} numerical, "
                    f"{len(self.categorical_features)} categorical, and "
                    f"{len(self.binary_features)} binary features")
        
        return {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features,
            'binary': self.binary_features
        }
    
    def _apply_transformations(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Apply preprocessing transformations."""
        X_transformed = X.copy()
        
        # Handle missing values using already fitted imputers
        if self.numerical_features and hasattr(self, 'numeric_imputer'):
            X_transformed[self.numerical_features] = self.numeric_imputer.transform(
                X_transformed[self.numerical_features]
            )
        
        if self.categorical_features and hasattr(self, 'categorical_imputer'):
            X_transformed[self.categorical_features] = self.categorical_imputer.transform(
                X_transformed[self.categorical_features]
            )
        
        # Encode categorical variables using already fitted encoders
        for feature in self.categorical_features:
            if feature in self.label_encoders:
                le = self.label_encoders[feature]
                # Handle unseen categories
                try:
                    X_transformed[feature] = le.transform(X_transformed[feature].astype(str))
                except ValueError:
                    # Handle unseen categories by mapping to most frequent class
                    known_classes = set(le.classes_)
                    X_transformed[feature] = X_transformed[feature].astype(str).apply(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    X_transformed[feature] = le.transform(X_transformed[feature])
        
        # Apply scaling/normalization if enabled using already fitted scaler
        if self.normalize_features and self.numerical_features:
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_transformed[self.numerical_features] = self.scaler.transform(
                    X_transformed[self.numerical_features]
                )
        
        return X_transformed
    
    def _calculate_data_quality_score(self, X: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if X.empty:
            return 0.0
        
        # Calculate missing value percentage
        missing_pct = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        # Calculate duplicate percentage
        duplicate_pct = X.duplicated().sum() / len(X)
        
        # Calculate quality score (higher is better)
        quality_score = 1.0 - (missing_pct * 0.5 + duplicate_pct * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    def _detect_outliers_iqr(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method."""
        if len(data) == 0:
            return np.array([], dtype=bool)
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessor state to file."""
        self.save_model(filepath)
    
    def save_model(self, filepath: str) -> None:
        """Save preprocessor state to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CreditDataPreprocessor':
        """Load preprocessor state from file."""
        import pickle
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor

    def get_feature_names(self) -> List[str]:
        """Get the names of transformed features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names or []


def load_and_preprocess_data(data_path: str, 
                           target_column: str = 'default_risk',
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess credit risk data for modeling.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
    target_column : str
        Name of the target variable column
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducible splits
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames and Series
        Preprocessed training and test sets
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data: {data.shape}")
    
    # Separate features and target
    if target_column in data.columns:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None
        logger.warning(f"Target column '{target_column}' not found. Proceeding without target.")
    
    # Initialize preprocessor
    preprocessor = CreditDataPreprocessor()
    
    # Validate data
    validation_results = preprocessor.validate_data(X)
    if any(validation_results.values()):
        logger.warning("Data validation issues found:")
        for issue_type, issues in validation_results.items():
            if issues:
                logger.warning(f"{issue_type}: {issues}")
    
    # Preprocess features
    X_processed = preprocessor.fit_transform(X)
    
    # Split data if target is available
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Default rate: {y_train.mean():.2%}")
        logger.info(f"Test set: {X_test.shape}, Default rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    else:
        logger.info(f"Processed data: {X_processed.shape}")
        return X_processed, None, None, None


def create_feature_summary(X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Create a comprehensive feature summary for analysis.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series, optional
        Target variable for correlation analysis
        
    Returns:
    --------
    summary : DataFrame
        Feature summary with statistics and insights
    """
    summary_data = []
    
    for feature in X.columns:
        feature_info = {
            'feature': feature,
            'dtype': str(X[feature].dtype),
            'missing_count': X[feature].isnull().sum(),
            'missing_pct': X[feature].isnull().sum() / len(X) * 100,
            'unique_values': X[feature].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(X[feature]):
            feature_info.update({
                'mean': X[feature].mean(),
                'std': X[feature].std(),
                'min': X[feature].min(),
                'max': X[feature].max(),
                'q25': X[feature].quantile(0.25),
                'q50': X[feature].quantile(0.50),
                'q75': X[feature].quantile(0.75)
            })
        else:
            # Categorical feature
            mode_value = X[feature].mode().iloc[0] if not X[feature].mode().empty else None
            feature_info.update({
                'mode': mode_value,
                'mode_frequency': (X[feature] == mode_value).sum() if mode_value else 0
            })
        
        # Correlation with target if available
        if y is not None and pd.api.types.is_numeric_dtype(X[feature]):
            try:
                correlation = X[feature].corr(y)
                feature_info['target_correlation'] = correlation
            except:
                feature_info['target_correlation'] = None
        
        summary_data.append(feature_info)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by target correlation if available
    if y is not None and 'target_correlation' in summary_df.columns:
        summary_df = summary_df.sort_values('target_correlation', key=abs, ascending=False)
    
    return summary_df


def main():
    """Demonstration of data preprocessing pipeline."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data first if needed
    try:
        from ..data.generate_sample_data import CreditDataGenerator
        
        print("Generating sample data for preprocessing demo...")
        generator = CreditDataGenerator()
        X, y = generator.generate_dataset(n_samples=500)
        
        # Save temporary data
        temp_data = X.copy()
        temp_data['default_risk'] = y
        temp_path = 'demos/credit_risk/data/temp_preprocessing_demo.csv'
        temp_data.to_csv(temp_path, index=False)
        
        print(f"Sample data saved to {temp_path}")
        
        # Load and preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess_data(temp_path)
        
        print(f"\nPreprocessing results:")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Create feature summary
        feature_summary = create_feature_summary(X_train, y_train)
        print(f"\nFeature summary:")
        print(feature_summary.head(10))
        
    except ImportError:
        print("Could not import data generator. Please ensure the generate_sample_data.py is available.")
    except Exception as e:
        print(f"Error in preprocessing demo: {e}")


if __name__ == "__main__":
    main()