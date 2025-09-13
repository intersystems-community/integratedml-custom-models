"""
Integration tests for fraud detection system.
Tests end-to-end workflows and component interactions.
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from demos.fraud_detection.data.generate_transaction_data import TransactionDataGenerator
from demos.fraud_detection.models.ensemble_fraud_detector import EnsembleFraudDetector
from demos.fraud_detection.features.realtime_features import RealTimeFeatureProcessor
from demos.fraud_detection.optimization.caching_strategies import FraudDetectionCacheManager


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end fraud detection workflow."""
    
    def test_complete_fraud_detection_pipeline(self, transaction_generator, test_config):
        """Test the complete fraud detection pipeline from data generation to prediction."""
        print("\n🔄 Testing Complete Fraud Detection Pipeline...")
        
        # Step 1: Generate transaction data
        print("   1️⃣ Generating transaction data...")
        transaction_data = transaction_generator.generate_transactions(
            n_transactions=test_config['test_data_size']
        )
        
        assert len(transaction_data) == test_config['test_data_size']
        assert 'is_fraud' in transaction_data.columns
        
        # Step 2: Initialize feature processor
        print("   2️⃣ Initializing feature processor...")
        feature_processor = RealTimeFeatureProcessor(
            enable_caching=True,
            cache_ttl_seconds=300
        )

        # Step 2.5: Fit the feature processor
        print("   2.5️⃣ Fitting feature processor...")
        feature_processor.fit(transaction_data)

        # Step 3: Process features for a sample of transactions
        print("   3️⃣ Processing transaction features...")
        sample_size = min(20, len(transaction_data))
        sample_data = transaction_data.head(sample_size)

        processed_features = []
        for idx, transaction in sample_data.iterrows():
            features = feature_processor.transform_single(transaction.to_dict())
            processed_features.append(features)
        
        features_df = pd.DataFrame(processed_features)
        assert len(features_df) == sample_size
        
        # Step 4: Initialize and train ensemble detector
        print("   4️⃣ Training ensemble detector...")
        ensemble_detector = EnsembleFraudDetector(
            combination_strategy='weighted_voting',
            weights={
                'rule_based': 0.25,
                'anomaly': 0.25,
                'neural': 0.25,
                'behavioral': 0.25
            }
        )
        
        # Simulate training with basic features
        X_train = features_df.fillna(0)
        y_train = sample_data['is_fraud'].values
        
        # Train the ensemble detector
        print("   4.5️⃣ Training ensemble detector...")
        ensemble_detector.fit(X_train, y_train)
        
        # Step 5: Make predictions
        print("   5️⃣ Making fraud predictions...")
        predictions = ensemble_detector.predict(X_train)
        
        assert len(predictions) == sample_size
        assert all(0 <= pred <= 1 for pred in predictions)
        
        # Step 6: Get detailed predictions with explanations
        print("   6️⃣ Generating detailed explanations...")
        detailed_results = ensemble_detector.predict_with_explanations(X_train)
        
        assert 'fraud_probability' in detailed_results
        assert 'risk_level' in detailed_results
        assert 'confidence' in detailed_results
        assert 'explanation' in detailed_results
        
        print("   ✅ Complete pipeline test successful!")
    
    def test_real_time_transaction_processing(self, transaction_generator):
        """Test real-time transaction processing workflow."""
        print("\n⚡ Testing Real-time Transaction Processing...")
        
        # Initialize components
        feature_processor = RealTimeFeatureProcessor(enable_caching=True)
        feature_processor.fit(train_data)
        ensemble_detector = EnsembleFraudDetector()
        
        # Train ensemble with small dataset for testing
        train_data = transaction_generator.generate_transactions(n_transactions=100)
        X_train = pd.DataFrame([
            feature_processor.transform_single(row.to_dict())
            for _, row in train_data.iterrows()
        ])
        y_train = train_data['is_fraud'].values
        ensemble_detector.fit(X_train.fillna(0), y_train)
        
        # Generate streaming transactions
        transaction_stream = transaction_generator.generate_transactions(
            n_transactions=50
        )
        
        processing_times = []
        predictions_made = 0
        
        # Process transactions one by one (simulating real-time)
        for idx, transaction in transaction_stream.iterrows():
            start_time = time.perf_counter()
            
            # Process features
            features = feature_processor.transform_single(
                transaction.to_dict()            )
            
            # Make prediction
            features_df = pd.DataFrame([features])
            prediction = ensemble_detector.predict(features_df.fillna(0))
            
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
            predictions_made += 1
            
            # Verify real-time performance
            assert processing_time_ms <= 200, f"Transaction {idx} processing time {processing_time_ms:.2f}ms too slow"
        
        # Analyze real-time performance
        avg_processing_time = np.mean(processing_times)
        p95_processing_time = np.percentile(processing_times, 95)
        
        print(f"   📊 Processed {predictions_made} transactions")
        print(f"   ⏱️ Average processing time: {avg_processing_time:.2f}ms")
        print(f"   ⏱️ P95 processing time: {p95_processing_time:.2f}ms")
        
        assert avg_processing_time <= 100, f"Average processing time {avg_processing_time:.2f}ms exceeds target"
        assert predictions_made == len(transaction_stream)
        
        print("   ✅ Real-time processing test successful!")
    
    def test_caching_integration(self, transaction_generator):
        """Test caching integration across components."""
        print("\n🗄️ Testing Caching Integration...")
        
        # Initialize with caching enabled
        feature_processor = RealTimeFeatureProcessor(enable_caching=True)
        cache_manager = FraudDetectionCacheManager()

        # Generate test transaction
        test_transaction = transaction_generator.generate_transactions(
            n_transactions=1
        ).iloc[0].to_dict()

        # Fit the feature processor with the transaction data
        feature_processor.fit(pd.DataFrame([test_transaction]))
        
        # First processing (cache miss)
        start_time = time.perf_counter()
        features_1 = feature_processor.transform_single(
            test_transaction        )
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Second processing (cache hit)
        start_time = time.perf_counter()
        features_2 = feature_processor.transform_single(
            test_transaction        )
        second_time = (time.perf_counter() - start_time) * 1000
        
        # Verify caching benefit
        speedup = first_time / second_time if second_time > 0 else 1
        print(f"   ⚡ Cache speedup: {speedup:.2f}x")
        print(f"   🕐 First processing: {first_time:.2f}ms")
        print(f"   🕐 Cached processing: {second_time:.2f}ms")
        
        # Features should be identical
        for key in features_1:
            if key in features_2:
                assert features_1[key] == features_2[key], f"Feature {key} differs between cached and non-cached"
        
        print("   ✅ Caching integration test successful!")
    
    def test_error_handling_and_resilience(self, transaction_generator):
        """Test error handling and system resilience."""
        print("\n🛡️ Testing Error Handling and Resilience...")
        
        ensemble_detector = EnsembleFraudDetector()
        feature_processor = RealTimeFeatureProcessor()
        
        # Test 1: Prediction on untrained model
        print("   🧪 Testing untrained model error handling...")
        test_features = pd.DataFrame({'amount': [100.0], 'hour_of_day': [14]})

        # The ensemble should handle untrained state gracefully
        try:
            predictions = ensemble_detector.predict(test_features)
            # If it doesn't raise an error, it should return some predictions
            if hasattr(predictions, '__len__'):
                assert len(predictions) == len(test_features)
            else:
                # Single prediction value
                assert isinstance(predictions, (int, float, np.number))
        except ValueError:
            # This is also acceptable behavior
            pass

        # Train the model for remaining tests
        print("   🧪 Training model for remaining tests...")
        train_data = transaction_generator.generate_transactions(n_transactions=50)
        X_train = pd.DataFrame([
            feature_processor.transform_single(row.to_dict())
            for _, row in train_data.iterrows()
        ])
        y_train = train_data['is_fraud'].values
        ensemble_detector.fit(X_train.fillna(0), y_train)
        
        # Empty dataframe
        empty_features = pd.DataFrame()
        try:
            predictions = ensemble_detector.predict(empty_features)
            assert len(predictions) == 0
        except (ValueError, IndexError):
            pass  # Expected behavior
        
        # Test 3: Feature processing with invalid transaction
        print("   🧪 Testing invalid transaction handling...")
        invalid_transaction = {
            'amount': -100,  # Invalid negative amount
            'transaction_timestamp': 'invalid_date',
            'customer_id': None
        }
        
        try:
            features = feature_processor.transform_single(
                invalid_transaction,
                historical_data=pd.DataFrame()
            )
            # Should handle gracefully and return default features
            assert isinstance(features, dict)
        except Exception as e:
            # Or raise appropriate error
            assert isinstance(e, (ValueError, TypeError))
        
        # Test 4: Valid predictions on normal data
        print("   🧪 Testing normal predictions...")
        test_features = pd.DataFrame({
            'amount': [100.0, 200.0],
            'hour_of_day': [14, 16],
            'merchant_risk_score': [0.3, 0.7]
        })

        predictions = ensemble_detector.predict(test_features)
        assert len(predictions) == 2
        assert all(0 <= p <= 1 for p in predictions)
        print("   ✅ Normal predictions working correctly")
        
        print("   ✅ Error handling test completed!")
    
    def test_data_consistency_across_components(self, transaction_generator):
        """Test data consistency across different components."""
        print("\n🔍 Testing Data Consistency...")
        
        # Generate test data
        test_data = transaction_generator.generate_transactions(n_transactions=10)
        
        # Initialize components
        feature_processor = RealTimeFeatureProcessor()
        feature_processor.fit(test_data)
        ensemble_detector = EnsembleFraudDetector()

        # Train with the test data
        X_train = pd.DataFrame([
            feature_processor.transform_single(row.to_dict())
            for _, row in test_data.iterrows()
        ])
        y_train = test_data['is_fraud'].values
        ensemble_detector.fit(X_train.fillna(0), y_train)
        
        # Process same transaction multiple times
        test_transaction = test_data.iloc[0].to_dict()
        
        features_list = []
        predictions_list = []
        
        for _ in range(5):
            # Process features
            features = feature_processor.transform_single(
                test_transaction,
                historical_data=pd.DataFrame()
            )
            features_list.append(features)
            
            # Make prediction
            features_df = pd.DataFrame([features]).fillna(0)
            prediction = ensemble_detector.predict(features_df)[0]
            predictions_list.append(prediction)
        
        # Verify consistency
        print(f"   📊 Feature consistency check...")
        first_features = features_list[0]
        for i, features in enumerate(features_list[1:], 1):
            for key in first_features:
                if key in features:
                    assert first_features[key] == features[key], f"Feature {key} inconsistent at iteration {i}"
        
        print(f"   📊 Prediction consistency check...")
        first_prediction = predictions_list[0]
        for i, prediction in enumerate(predictions_list[1:], 1):
            assert abs(prediction - first_prediction) < 1e-6, f"Prediction inconsistent at iteration {i}"
        
        print("   ✅ Data consistency test successful!")
    
    def test_scalability_integration(self, transaction_generator):
        """Test system scalability with increasing load."""
        print("\n📈 Testing Scalability Integration...")
        
        # Initialize system
        feature_processor = RealTimeFeatureProcessor(enable_caching=True)
        feature_processor.fit(train_data)
        ensemble_detector = EnsembleFraudDetector()

        # Train with small dataset
        train_data = transaction_generator.generate_transactions(n_transactions=50)
        X_train = pd.DataFrame([
            feature_processor.transform_single(row.to_dict())
            for _, row in train_data.iterrows()
        ])
        y_train = train_data['is_fraud'].values
        ensemble_detector.fit(X_train.fillna(0), y_train)
        
        # Test with increasing data sizes
        data_sizes = [10, 50, 100, 200]
        performance_metrics = []
        
        for size in data_sizes:
            print(f"   📊 Testing with {size} transactions...")
            
            # Generate data
            test_data = transaction_generator.generate_transactions(n_transactions=size)
            
            # Process all transactions
            start_time = time.perf_counter()
            
            processed_count = 0
            for idx, transaction in test_data.iterrows():
                # Process features
                features = feature_processor.transform_single(
                    transaction.to_dict()                )
                
                # Make prediction
                features_df = pd.DataFrame([features]).fillna(0)
                prediction = ensemble_detector.predict(features_df)
                processed_count += 1
            
            total_time = time.perf_counter() - start_time
            avg_time_per_transaction = (total_time / size) * 1000  # ms
            throughput = size / total_time  # transactions per second
            
            performance_metrics.append({
                'size': size,
                'total_time': total_time,
                'avg_time_ms': avg_time_per_transaction,
                'throughput_tps': throughput
            })
            
            print(f"      ⏱️ Average time per transaction: {avg_time_per_transaction:.2f}ms")
            print(f"      🚀 Throughput: {throughput:.1f} TPS")
            
            # Verify acceptable performance
            assert avg_time_per_transaction <= 200, f"Performance degraded at size {size}"
            assert processed_count == size
        
        # Analyze scalability
        print(f"\n   📈 Scalability Analysis:")
        for metrics in performance_metrics:
            print(f"      {metrics['size']} txn: {metrics['avg_time_ms']:.2f}ms avg, {metrics['throughput_tps']:.1f} TPS")
        
        print("   ✅ Scalability integration test successful!")


@pytest.mark.integration 
class TestComponentInteractions:
    """Test specific component interactions and interfaces."""
    
    def test_feature_processor_ensemble_integration(self, sample_single_transaction):
        """Test integration between feature processor and ensemble detector."""
        feature_processor = RealTimeFeatureProcessor()
        ensemble_detector = EnsembleFraudDetector()

        # Train with small dataset
        from demos.fraud_detection.data.generate_transaction_data import TransactionDataGenerator
        generator = TransactionDataGenerator()
        train_data = generator.generate_transactions(n_transactions=20)
        feature_processor.fit(train_data)
        X_train = pd.DataFrame([
            feature_processor.transform_single(row.to_dict())
            for _, row in train_data.iterrows()
        ])
        y_train = train_data['is_fraud'].values
        ensemble_detector.fit(X_train.fillna(0), y_train)
        
        # Process features
        features = feature_processor.transform_single(
            sample_single_transaction        )
        
        # Convert to DataFrame for ensemble
        features_df = pd.DataFrame([features]).fillna(0)
        
        # Make prediction
        prediction = ensemble_detector.predict(features_df)
        
        assert len(prediction) == 1
        assert 0 <= prediction[0] <= 1
    
    def test_cache_manager_integration(self):
        """Test cache manager integration with other components."""
        cache_manager = FraudDetectionCacheManager()

        # Test customer profile caching
        customer_id = "test_customer_123"
        profile = {"age": 30, "credit_score": 750}

        # Cache customer profile
        success = cache_manager.cache_customer_profile(customer_id, profile)
        assert success

        # Retrieve customer profile
        cached_profile = cache_manager.get_customer_profile(customer_id)
        assert cached_profile == profile

        # Test feature caching
        feature_key = "transaction_features_456"
        features = {"amount": 100.0, "velocity": 0.3}
        cache_manager.feature_cache.put(feature_key, features)

        cached_features = cache_manager.feature_cache.get(feature_key)
        assert cached_features == features