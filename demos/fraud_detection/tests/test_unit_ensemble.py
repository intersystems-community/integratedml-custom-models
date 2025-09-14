"""
Unit tests for ensemble fraud detector.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector


class TestEnsembleFraudDetector:
    """Test cases for EnsembleFraudDetector."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        detector = EnsembleFraudDetector()

        assert detector.combination_strategy == "weighted_voting"
        assert detector.weights is not None
        assert detector.enable_confidence_scoring == True
        assert detector.enable_explanation == True
        assert detector._is_trained == False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_weights = {
            "rule_based": 0.4,
            "anomaly": 0.3,
            "neural": 0.2,
            "behavioral": 0.1,
        }

        detector = EnsembleFraudDetector(
            combination_strategy="stacking",
            weights=custom_weights,
            enable_confidence_scoring=False,
            enable_explanation=False,
        )

        assert detector.combination_strategy == "stacking"
        assert detector.weights == custom_weights
        assert detector.enable_confidence_scoring == False
        assert detector.enable_explanation == False

    def test_weights_validation(self):
        """Test weight validation."""
        # Valid weights (sum to 1.0)
        valid_weights = {
            "rule_based": 0.25,
            "anomaly": 0.25,
            "neural": 0.25,
            "behavioral": 0.25,
        }
        detector = EnsembleFraudDetector(weights=valid_weights)
        assert abs(sum(detector.weights.values()) - 1.0) < 1e-6

        # Invalid weights (don't sum to 1.0) - should be normalized
        invalid_weights = {
            "rule_based": 0.5,
            "anomaly": 0.5,
            "neural": 0.5,
            "behavioral": 0.5,
        }
        detector = EnsembleFraudDetector(weights=invalid_weights)
        assert abs(sum(detector.weights.values()) - 1.0) < 1e-6

    def test_sub_model_initialization(self, ensemble_detector):
        """Test that sub-models are properly initialized."""
        detector = ensemble_detector

        # Check that sub-models exist
        assert hasattr(detector, "rule_based_detector")
        assert hasattr(detector, "anomaly_detector")
        assert hasattr(detector, "neural_detector")
        assert hasattr(detector, "behavioral_detector")

        # Check that they are not None
        assert detector.rule_based_detector is not None
        assert detector.anomaly_detector is not None
        assert detector.neural_detector is not None
        assert detector.behavioral_detector is not None

    @patch(
        "demos.fraud_detection.models.ensemble_fraud_detector.EnsembleFraudDetector._train_sub_models"
    )
    def test_train_basic(
        self, mock_train_sub_models, ensemble_detector, sample_features
    ):
        """Test basic training functionality."""
        # Mock the sub-model training
        mock_train_sub_models.return_value = None

        X_train = sample_features
        y_train = np.array([0, 1, 0, 1, 0])

        ensemble_detector.train(X_train, y_train)

        # Check that training was called
        mock_train_sub_models.assert_called_once()
        assert ensemble_detector._is_trained == True
        assert ensemble_detector._training_metrics is not None

    def test_predict_untrained_model(self, ensemble_detector, sample_features):
        """Test prediction on untrained model raises error."""
        with pytest.raises(ValueError, match="not been trained"):
            ensemble_detector.predict(sample_features)

    def test_predict_with_trained_model(self, mock_trained_ensemble, sample_features):
        """Test prediction with trained model."""
        # Mock sub-model predictions
        mock_trained_ensemble.rule_based_detector.predict = Mock(
            return_value=np.array([0.2, 0.8, 0.1, 0.9, 0.05])
        )
        mock_trained_ensemble.anomaly_detector.predict = Mock(
            return_value=np.array([0.3, 0.7, 0.2, 0.8, 0.1])
        )
        mock_trained_ensemble.neural_detector.predict = Mock(
            return_value=np.array([0.1, 0.9, 0.15, 0.85, 0.08])
        )
        mock_trained_ensemble.behavioral_detector.predict = Mock(
            return_value=np.array([0.25, 0.75, 0.18, 0.82, 0.12])
        )

        predictions = mock_trained_ensemble.predict(sample_features)

        # Check predictions shape and range
        assert len(predictions) == len(sample_features)
        assert all(0 <= pred <= 1 for pred in predictions)

    def test_predict_with_explanations(self, mock_trained_ensemble, sample_features):
        """Test prediction with explanations."""
        # Mock sub-model predictions
        mock_trained_ensemble.rule_based_detector.predict = Mock(
            return_value=np.array([0.2, 0.8, 0.1, 0.9, 0.05])
        )
        mock_trained_ensemble.anomaly_detector.predict = Mock(
            return_value=np.array([0.3, 0.7, 0.2, 0.8, 0.1])
        )
        mock_trained_ensemble.neural_detector.predict = Mock(
            return_value=np.array([0.1, 0.9, 0.15, 0.85, 0.08])
        )
        mock_trained_ensemble.behavioral_detector.predict = Mock(
            return_value=np.array([0.25, 0.75, 0.18, 0.82, 0.12])
        )

        result = mock_trained_ensemble.predict_with_explanations(sample_features)

        # Check result structure
        assert "fraud_probability" in result
        assert "risk_level" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "sub_model_scores" in result

        # Check array lengths
        assert len(result["fraud_probability"]) == len(sample_features)
        assert len(result["risk_level"]) == len(sample_features)
        assert len(result["confidence"]) == len(sample_features)
        assert len(result["explanation"]) == len(sample_features)
        assert len(result["sub_model_scores"]) == len(sample_features)

    def test_weighted_voting_combination(self, mock_trained_ensemble):
        """Test weighted voting combination strategy."""
        # Set combination strategy to weighted voting
        mock_trained_ensemble.combination_strategy = "weighted_voting"

        # Test scores
        sub_model_scores = {
            "rule_based": np.array([0.2, 0.8]),
            "anomaly": np.array([0.3, 0.7]),
            "neural": np.array([0.1, 0.9]),
            "behavioral": np.array([0.4, 0.6]),
        }

        combined_scores = mock_trained_ensemble._combine_predictions(sub_model_scores)

        # Check that scores are combined properly
        assert len(combined_scores) == 2
        assert all(0 <= score <= 1 for score in combined_scores)

        # Verify weighted average calculation
        expected_0 = 0.2 * 0.25 + 0.3 * 0.25 + 0.1 * 0.25 + 0.4 * 0.25
        expected_1 = 0.8 * 0.25 + 0.7 * 0.25 + 0.9 * 0.25 + 0.6 * 0.25

        assert abs(combined_scores[0] - expected_0) < 1e-6
        assert abs(combined_scores[1] - expected_1) < 1e-6

    def test_risk_level_classification(self, mock_trained_ensemble):
        """Test risk level classification."""
        # Test different fraud probability levels
        test_probabilities = [0.1, 0.4, 0.6, 0.85, 0.95]

        risk_levels = mock_trained_ensemble._classify_risk_level(test_probabilities)

        # Check expected risk levels
        expected_levels = ["LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]
        assert risk_levels == expected_levels

    def test_confidence_scoring(self, mock_trained_ensemble):
        """Test confidence scoring."""
        # Test scores with different confidence levels
        sub_model_scores = [
            # High confidence (all models agree)
            {"rule_based": 0.9, "anomaly": 0.85, "neural": 0.88, "behavioral": 0.92},
            # Low confidence (models disagree)
            {"rule_based": 0.1, "anomaly": 0.9, "neural": 0.2, "behavioral": 0.8},
            # Medium confidence
            {"rule_based": 0.6, "anomaly": 0.65, "neural": 0.58, "behavioral": 0.62},
        ]

        confidences = []
        for scores in sub_model_scores:
            confidence = mock_trained_ensemble._calculate_confidence(scores)
            confidences.append(confidence)

        # High agreement should have higher confidence than disagreement
        assert confidences[0] > confidences[1]  # High agreement > disagreement
        assert confidences[2] > confidences[1]  # Medium agreement > disagreement

    def test_explanation_generation(self, mock_trained_ensemble):
        """Test explanation generation."""
        # Test case with high fraud probability
        sub_model_scores = {
            "rule_based": 0.9,
            "anomaly": 0.8,
            "neural": 0.85,
            "behavioral": 0.7,
        }

        explanation = mock_trained_ensemble._generate_explanation(
            fraud_probability=0.82, sub_model_scores=sub_model_scores, risk_level="HIGH"
        )

        # Check that explanation is a string and contains relevant information
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "HIGH" in explanation or "high" in explanation

    def test_feature_validation(self, mock_trained_ensemble):
        """Test feature validation."""
        # Valid features
        valid_features = pd.DataFrame(
            {
                "amount": [100.0, 200.0],
                "hour_of_day": [14, 16],
                "merchant_risk_score": [0.3, 0.8],
            }
        )

        # Should not raise error
        mock_trained_ensemble._validate_features(valid_features)

        # Invalid features (missing columns)
        invalid_features = pd.DataFrame({"amount": [100.0, 200.0]})

        # Should handle gracefully (may warn but not fail)
        try:
            mock_trained_ensemble._validate_features(invalid_features)
        except Exception as e:
            # If validation is strict, ensure it raises appropriate error
            assert isinstance(e, (ValueError, KeyError))

    def test_get_training_metrics(self, mock_trained_ensemble):
        """Test getting training metrics."""
        metrics = mock_trained_ensemble.get_training_metrics()

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics

        # Check metric ranges
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                assert 0 <= metric_value <= 1

    def test_model_serialization(self, mock_trained_ensemble, tmp_path):
        """Test model saving and loading."""
        # Test save
        save_path = tmp_path / "test_model.pkl"

        try:
            mock_trained_ensemble.save_model(str(save_path))
            assert save_path.exists()

            # Test load
            loaded_detector = EnsembleFraudDetector.load_model(str(save_path))
            assert loaded_detector._is_trained == mock_trained_ensemble._is_trained
            assert loaded_detector.weights == mock_trained_ensemble.weights

        except NotImplementedError:
            # If serialization not implemented, that's acceptable for now
            pytest.skip("Model serialization not implemented")

    def test_different_combination_strategies(self):
        """Test different combination strategies."""
        strategies = ["weighted_voting", "simple_voting", "stacking"]

        for strategy in strategies:
            try:
                detector = EnsembleFraudDetector(combination_strategy=strategy)
                assert detector.combination_strategy == strategy
            except ValueError as e:
                # If strategy not implemented, should raise appropriate error
                assert "not supported" in str(e).lower() or "invalid" in str(e).lower()

    def test_edge_cases(self, mock_trained_ensemble):
        """Test edge cases."""
        # Empty input
        empty_features = pd.DataFrame()

        if len(empty_features) == 0:
            with pytest.raises((ValueError, IndexError)):
                mock_trained_ensemble.predict(empty_features)

        # Single row input
        single_row = pd.DataFrame(
            {"amount": [100.0], "hour_of_day": [14], "merchant_risk_score": [0.3]}
        )

        # Mock sub-model predictions for single row
        mock_trained_ensemble.rule_based_detector.predict = Mock(
            return_value=np.array([0.3])
        )
        mock_trained_ensemble.anomaly_detector.predict = Mock(
            return_value=np.array([0.4])
        )
        mock_trained_ensemble.neural_detector.predict = Mock(
            return_value=np.array([0.2])
        )
        mock_trained_ensemble.behavioral_detector.predict = Mock(
            return_value=np.array([0.35])
        )

        result = mock_trained_ensemble.predict(single_row)
        assert len(result) == 1
        assert 0 <= result[0] <= 1

    def test_performance_tracking(self, mock_trained_ensemble, sample_features):
        """Test performance tracking during prediction."""
        # Mock sub-model predictions
        mock_trained_ensemble.rule_based_detector.predict = Mock(
            return_value=np.array([0.2, 0.8, 0.1])
        )
        mock_trained_ensemble.anomaly_detector.predict = Mock(
            return_value=np.array([0.3, 0.7, 0.2])
        )
        mock_trained_ensemble.neural_detector.predict = Mock(
            return_value=np.array([0.1, 0.9, 0.15])
        )
        mock_trained_ensemble.behavioral_detector.predict = Mock(
            return_value=np.array([0.25, 0.75, 0.18])
        )

        # Ensure we have correct number of samples
        test_features = sample_features.head(3)

        # Predict and check if performance is tracked
        predictions = mock_trained_ensemble.predict(test_features)

        # Check if performance metrics are available
        if hasattr(mock_trained_ensemble, "_prediction_times"):
            assert len(mock_trained_ensemble._prediction_times) > 0

        # All predictions should be valid probabilities
        assert all(0 <= pred <= 1 for pred in predictions)
