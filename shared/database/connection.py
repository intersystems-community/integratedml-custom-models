"""
IRIS Database Connection Module

Provides connection management and utilities for InterSystems IRIS database
with IntegratedML support using native IRIS connectivity.
"""

import os
import logging
import subprocess
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

try:
    import iris
    IRIS_AVAILABLE = True
    logger.info("IRIS Python package is available")
except ImportError:
    IRIS_AVAILABLE = False
    logger.warning("IRIS Python package not available. Using HTTP-based approach for basic operations.")


class IRISConnection:
    """
    Manages connections to InterSystems IRIS database with IntegratedML support.
    Uses native IRIS connections when available, HTTP API as fallback.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        namespace: str = None,
        use_ssl: bool = False,
        connection_timeout: int = 30,
        max_connections: int = 10
    ):
        """
        Initialize IRIS connection parameters.
        
        Args:
            host: IRIS database host
            port: IRIS database port (1972 for IRIS protocol)
            username: Database username
            password: Database password
            namespace: IRIS namespace
            use_ssl: Whether to use SSL connection
            connection_timeout: Connection timeout in seconds
            max_connections: Maximum number of connections in pool
        """
        self.host = host or os.getenv('IRIS_HOST', 'localhost')
        self.port = port or int(os.getenv('IRIS_PORT', '1972'))
        self.username = username or os.getenv('IRIS_USERNAME', 'demo')
        self.password = password or os.getenv('IRIS_PASSWORD', 'demo')
        self.namespace = namespace or os.getenv('IRIS_NAMESPACE', 'USER')
        self.use_ssl = use_ssl or os.getenv('IRIS_USE_SSL', 'false').lower() == 'true'
        self.connection_timeout = connection_timeout
        self.max_connections = max_connections
        
        self._native_connection = None
        self._http_session = None
        
        logger.info(f"Initialized IRIS connection to {self.host}:{self.port}/{self.namespace}")
    
    def get_native_connection(self):
        """Get native IRIS connection for IntegratedML operations."""
        if not IRIS_AVAILABLE:
            raise RuntimeError("Native IRIS package not available. Install 'intersystems-iris' package.")
            
        if self._native_connection is None:
            try:
                connection_params = {
                    'hostname': self.host,
                    'port': self.port,
                    'namespace': self.namespace,
                    'username': self.username,
                    'password': self.password
                }
                
                self._native_connection = iris.connect(**connection_params)
                logger.info("Established native IRIS connection")
            except Exception as e:
                logger.error(f"Failed to create native IRIS connection: {e}")
                raise
        return self._native_connection
    
    def execute_sql(self, sql: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query using native IRIS connection.
        
        Args:
            sql: SQL query to execute
            parameters: Query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        if IRIS_AVAILABLE:
            return self._execute_native_sql(sql, parameters)
        else:
            return self._execute_http_sql(sql, parameters)
    
    def _execute_native_sql(self, sql: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SQL using native IRIS connection."""
        try:
            conn = self.get_native_connection()
            iris_obj = conn.cursor()
            
            if parameters:
                iris_obj.execute(sql, parameters)
            else:
                iris_obj.execute(sql)
            
            # Get column names
            if iris_obj.description:
                columns = [desc[0] for desc in iris_obj.description]
                rows = iris_obj.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            else:
                # No results (e.g., INSERT, UPDATE, DELETE)
                return []
                
        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            raise
    
    def _execute_http_sql(self, sql: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SQL using HTTP REST API fallback."""
        import requests
        import json
        
        try:
            # IRIS REST API endpoint for SQL execution
            web_port = os.getenv('IRIS_WEB_PORT', 52776)
            base_url = f"http://{self.host}:{web_port}"
            
            # Prepare SQL with parameters
            if parameters:
                # Simple parameter substitution for basic cases
                for key, value in parameters.items():
                    if isinstance(value, str):
                        sql = sql.replace(f":{key}", f"'{value}'")
                    else:
                        sql = sql.replace(f":{key}", str(value))
            
            # Use IRIS REST SQL interface
            url = f"{base_url}/api/atelier/v1/{self.namespace}/action/query"
            
            payload = {
                "query": sql,
                "parameters": []
            }
            
            auth = (self.username, self.password)
            response = requests.post(url, json=payload, auth=auth, timeout=self.connection_timeout)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('result', {}).get('content', [])
            else:
                logger.error(f"HTTP SQL execution failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to execute HTTP SQL query: {e}")
            # Return empty result rather than failing completely
            return []
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        if IRIS_AVAILABLE:
            conn = self.get_native_connection()
            try:
                conn.begin()
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
        else:
            # For HTTP fallback, each statement is auto-committed
            logger.warning("Transaction context not available with HTTP fallback")
            yield None
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.execute_sql("SELECT 1 as test")
            return len(result) > 0 and result[0].get('test') == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_integratedml_info(self) -> Dict[str, Any]:
        """
        Get IntegratedML configuration and status information.
        
        Returns:
            Dictionary with IntegratedML information
        """
        try:
            info = {}
            
            # Check if IntegratedML is enabled
            result = self.execute_sql("SELECT $SYSTEM.SQL.GetMLConfig() as ml_config")
            info['ml_enabled'] = bool(result[0]['ml_config']) if result else False
            
            # Get available ML providers
            providers_query = """
            SELECT provider_name, provider_type, is_available
            FROM INFORMATION_SCHEMA.ML_PROVIDERS
            """
            try:
                providers = self.execute_sql(providers_query)
                info['providers'] = providers
            except:
                info['providers'] = []
            
            # Get model count
            try:
                models_result = self.execute_sql("SELECT COUNT(*) FROM INFORMATION_SCHEMA.ML_MODELS")
                info['model_count'] = models_result[0]['COUNT'] if models_result else 0
            except:
                info['model_count'] = 0
                
            return info
        except Exception as e:
            logger.error(f"Failed to get IntegratedML info: {e}")
            return {'error': str(e)}
    
    def create_integratedml_model(
        self,
        model_name: str,
        model_type: str,
        features: List[str],
        target: str,
        table_name: str
    ) -> bool:
        """
        Create an IntegratedML model.
        
        Args:
            model_name: Name of the model to create
            model_type: Type of model (e.g., 'RandomForest', 'LogisticRegression')
            features: List of feature column names
            target: Target column name
            table_name: Name of the source table
            
        Returns:
            True if successful, False otherwise
        """
        try:
            features_str = ', '.join(features)
            
            # Create model SQL
            sql = f"""
            CREATE MODEL {model_name}
            PREDICTING ({target})
            FROM {table_name}
            SELECT {features_str}, {target}
            USING {model_type}
            """
            
            self.execute_sql(sql)
            logger.info(f"Created IntegratedML model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return False
    
    def train_integratedml_model(self, model_name: str) -> bool:
        """
        Train an IntegratedML model.
        
        Args:
            model_name: Name of the model to train
            
        Returns:
            True if successful, False otherwise
        """
        try:
            sql = f"TRAIN MODEL {model_name}"
            self.execute_sql(sql)
            logger.info(f"Training started for model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            return False
    
    def predict_with_model(
        self,
        model_name: str,
        input_data: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Make predictions using an IntegratedML model.
        
        Args:
            model_name: Name of the trained model
            input_data: Dictionary of feature values
            
        Returns:
            Prediction result or None if failed
        """
        try:
            # Build prediction SQL
            features_list = ', '.join(f'{v} as {k}' for k, v in input_data.items())
            
            sql = f"""
            SELECT PREDICT({model_name}) as prediction
            FROM (SELECT {features_list})
            """
            
            result = self.execute_sql(sql)
            if result and len(result) > 0:
                return result[0].get('prediction')
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to make prediction with model {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an IntegratedML model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            sql = f"SELECT * FROM INFORMATION_SCHEMA.ML_MODELS WHERE MODEL_NAME = '{model_name}'"
            result = self.execute_sql(sql)
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    def close(self):
        """Close database connections."""
        if self._native_connection and IRIS_AVAILABLE:
            try:
                self._native_connection.close()
                self._native_connection = None
                logger.info("Closed native IRIS connection")
            except Exception as e:
                logger.warning(f"Error closing native connection: {e}")
        
        if self._http_session:
            try:
                self._http_session.close()
                self._http_session = None
                logger.info("Closed HTTP session")
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {e}")


# Global connection instance
_global_connection: Optional[IRISConnection] = None


def get_connection() -> IRISConnection:
    """Get global IRIS connection instance."""
    global _global_connection
    if _global_connection is None:
        _global_connection = IRISConnection()
    return _global_connection


def test_connection() -> bool:
    """Test global database connection."""
    try:
        conn = get_connection()
        return conn.test_connection()
    except Exception as e:
        logger.error(f"Global connection test failed: {e}")
        return False


def close_connection():
    """Close global database connection."""
    global _global_connection
    if _global_connection:
        _global_connection.close()
        _global_connection = None