"""
IntegratedML Model Management Module

Handles model deployment, training, and lifecycle management in IRIS IntegratedML.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from .connection import IRISConnection, get_connection

# Configure logging
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models in IRIS IntegratedML environment."""

    def __init__(self, connection: IRISConnection = None):
        """
        Initialize model manager.

        Args:
            connection: IRIS connection instance
        """
        self.conn = connection or get_connection()
        self.model_path = Path(os.getenv("IML_MODEL_PATH", "/app/models"))
        self.model_path.mkdir(parents=True, exist_ok=True)

    def create_model(
        self,
        model_name: str,
        model_type: str,
        demo_category: str,
        training_table: str,
        target_column: str,
        feature_columns: List[str],
        model_parameters: Dict[str, Any] = None,
    ) -> bool:
        """
        Create and train a model in IntegratedML.

        Args:
            model_name: Name of the model
            model_type: Type of model (Classification, Regression, etc.)
            demo_category: Demo category (CreditRisk, FraudDetection, etc.)
            training_table: Table containing training data
            target_column: Target column for prediction
            feature_columns: List of feature columns
            model_parameters: Optional model parameters

        Returns:
            True if model creation successful, False otherwise
        """
        try:
            # Register model in registry
            self._register_model(model_name, model_type, demo_category)

            # Prepare feature list for SQL
            features_sql = ", ".join(feature_columns)

            # Create model SQL based on type
            if model_type.lower() == "classification":
                create_sql = f"""
                CREATE MODEL {model_name}
                PREDICTING ({target_column})
                FROM {training_table}
                SELECT {features_sql}, {target_column}
                """
            elif model_type.lower() == "regression":
                create_sql = f"""
                CREATE MODEL {model_name}
                PREDICTING ({target_column})
                FROM {training_table}
                SELECT {features_sql}, {target_column}
                """
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Add model parameters if provided
            if model_parameters:
                params_sql = self._format_model_parameters(model_parameters)
                create_sql += f" WITH {params_sql}"

            # Execute model creation
            self.conn.execute_integratedml(create_sql)

            # Update model status
            self._update_model_status(model_name, "TRAINING")

            logger.info(f"Created model {model_name} in IntegratedML")
            return True

        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            self._update_model_status(model_name, "ERROR")
            return False

    def train_model(self, model_name: str) -> bool:
        """
        Train an existing model.

        Args:
            model_name: Name of the model to train

        Returns:
            True if training successful, False otherwise
        """
        try:
            # Check if model exists
            if not self._model_exists(model_name):
                logger.error(f"Model {model_name} does not exist")
                return False

            # Train the model
            train_sql = f"TRAIN MODEL {model_name}"
            self.conn.execute_integratedml(train_sql)

            # Update status
            self._update_model_status(model_name, "ACTIVE")

            logger.info(f"Trained model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            self._update_model_status(model_name, "ERROR")
            return False

    def predict(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        return_probability: bool = False,
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model
            input_data: Input data for prediction
            return_probability: Whether to return prediction probabilities

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Validate model is active
            model_info = self.get_model_info(model_name)
            if not model_info or model_info.get("status") != "ACTIVE":
                raise ValueError(f"Model {model_name} is not active")

            # Prepare input data for SQL
            columns = list(input_data.keys())
            values = list(input_data.values())

            # Build prediction SQL
            select_clause = f"PREDICT({model_name}) AS prediction"
            if return_probability:
                select_clause += f", PROBABILITY({model_name}) AS probability"

            values_clause = ", ".join(
                [f"'{v}'" if isinstance(v, str) else str(v) for v in values]
            )
            columns_clause = ", ".join(columns)

            predict_sql = f"""
            SELECT {select_clause}
            FROM (SELECT {values_clause}) AS InputData({columns_clause})
            """

            # Execute prediction
            start_time = datetime.now()
            result = self.conn.execute_integratedml(predict_sql)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds() * 1000

            # Process results
            if result and len(result) > 0:
                prediction_result = {
                    "model_name": model_name,
                    "prediction": result[0][0],
                    "execution_time_ms": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }

                if return_probability and len(result[0]) > 1:
                    prediction_result["probability"] = result[0][1]

                # Log prediction
                self._log_prediction(
                    model_name, input_data, prediction_result, execution_time
                )

                return prediction_result
            else:
                raise ValueError("No prediction result returned")

        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return {"error": str(e)}

    def evaluate_model(self, model_name: str, test_table: str = None) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model_name: Name of the model
            test_table: Optional test table for evaluation

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if test_table:
                # Evaluate on specific test table
                eval_sql = f"""
                SELECT 
                    ACCURACY({model_name}) AS accuracy,
                    PRECISION({model_name}) AS precision,
                    RECALL({model_name}) AS recall,
                    F1_SCORE({model_name}) AS f1_score
                FROM {test_table}
                """
            else:
                # Use built-in validation
                eval_sql = f"""
                SELECT 
                    * 
                FROM INFORMATION_SCHEMA.ML_VALIDATION_METRICS
                WHERE MODEL_NAME = '{model_name}'
                """

            result = self.conn.execute_integratedml(eval_sql)

            if result:
                if test_table:
                    return {
                        "model_name": model_name,
                        "accuracy": float(result[0][0]) if result[0][0] else None,
                        "precision": float(result[0][1]) if result[0][1] else None,
                        "recall": float(result[0][2]) if result[0][2] else None,
                        "f1_score": float(result[0][3]) if result[0][3] else None,
                        "evaluation_time": datetime.now().isoformat(),
                    }
                else:
                    # Parse validation metrics result
                    return {
                        "model_name": model_name,
                        "validation_metrics": result,
                        "evaluation_time": datetime.now().isoformat(),
                    }
            else:
                return {"error": "No evaluation results available"}

        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            return {"error": str(e)}

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information or None if not found
        """
        try:
            # Get model from registry
            registry_sql = """
            SELECT model_name, model_type, demo_category, status, 
                   created_at, updated_at, metadata, performance_metrics
            FROM ModelRegistry
            WHERE model_name = ?
            """

            result = self.conn.execute_query(registry_sql, {"model_name": model_name})

            if result:
                row = result[0]
                model_info = {
                    "model_name": row[0],
                    "model_type": row[1],
                    "demo_category": row[2],
                    "status": row[3],
                    "created_at": str(row[4]),
                    "updated_at": str(row[5]),
                    "metadata": json.loads(row[6]) if row[6] else None,
                    "performance_metrics": json.loads(row[7]) if row[7] else None,
                }

                # Get IntegratedML model info if available
                try:
                    ml_info_sql = f"""
                    SELECT * FROM INFORMATION_SCHEMA.ML_MODELS
                    WHERE MODEL_NAME = '{model_name}'
                    """
                    ml_result = self.conn.execute_integratedml(ml_info_sql)
                    if ml_result:
                        model_info["integratedml_info"] = ml_result[0]
                except:
                    pass

                return model_info

            return None

        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None

    def list_models(self, demo_category: str = None) -> List[Dict[str, Any]]:
        """
        List all models or models for a specific demo category.

        Args:
            demo_category: Optional demo category filter

        Returns:
            List of model information dictionaries
        """
        try:
            if demo_category:
                sql = """
                SELECT model_name, model_type, demo_category, status, updated_at
                FROM ModelRegistry
                WHERE demo_category = ?
                ORDER BY updated_at DESC
                """
                result = self.conn.execute_query(sql, {"demo_category": demo_category})
            else:
                sql = """
                SELECT model_name, model_type, demo_category, status, updated_at
                FROM ModelRegistry
                ORDER BY updated_at DESC
                """
                result = self.conn.execute_query(sql)

            models = []
            for row in result:
                models.append(
                    {
                        "model_name": row[0],
                        "model_type": row[1],
                        "demo_category": row[2],
                        "status": row[3],
                        "updated_at": str(row[4]),
                    }
                )

            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from IntegratedML and registry.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete from IntegratedML
            try:
                drop_sql = f"DROP MODEL {model_name}"
                self.conn.execute_integratedml(drop_sql)
            except:
                pass  # Model might not exist in IntegratedML

            # Delete from registry
            delete_sql = "DELETE FROM ModelRegistry WHERE model_name = ?"
            self.conn.execute_query(delete_sql, {"model_name": model_name})

            logger.info(f"Deleted model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def _register_model(self, model_name: str, model_type: str, demo_category: str):
        """Register model in the model registry."""
        insert_sql = """
        INSERT OR REPLACE INTO ModelRegistry 
        (model_name, model_type, demo_category, status, updated_at)
        VALUES (?, ?, ?, 'CREATING', CURRENT_TIMESTAMP)
        """
        self.conn.execute_query(
            insert_sql,
            {
                "model_name": model_name,
                "model_type": model_type,
                "demo_category": demo_category,
            },
        )

    def _update_model_status(self, model_name: str, status: str):
        """Update model status in registry."""
        update_sql = """
        UPDATE ModelRegistry 
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE model_name = ?
        """
        self.conn.execute_query(
            update_sql, {"status": status, "model_name": model_name}
        )

    def _model_exists(self, model_name: str) -> bool:
        """Check if model exists in registry."""
        check_sql = "SELECT COUNT(*) FROM ModelRegistry WHERE model_name = ?"
        result = self.conn.execute_query(check_sql, {"model_name": model_name})
        return result[0][0] > 0 if result else False

    def _format_model_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format model parameters for SQL."""
        param_strings = []
        for key, value in parameters.items():
            if isinstance(value, str):
                param_strings.append(f"{key}='{value}'")
            else:
                param_strings.append(f"{key}={value}")
        return ", ".join(param_strings)

    def _log_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        execution_time: float,
    ):
        """Log prediction to the predictions table."""
        try:
            log_sql = """
            INSERT INTO ModelPredictions 
            (model_name, input_data, prediction_result, confidence_score, execution_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """

            confidence = prediction_result.get("probability", 0.0)
            if isinstance(confidence, dict):
                confidence = max(confidence.values()) if confidence else 0.0

            self.conn.execute_query(
                log_sql,
                {
                    "model_name": model_name,
                    "input_data": json.dumps(input_data),
                    "prediction_result": json.dumps(prediction_result),
                    "confidence_score": float(confidence),
                    "execution_time_ms": int(execution_time),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")
