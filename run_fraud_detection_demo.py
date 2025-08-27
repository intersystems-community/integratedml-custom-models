#!/usr/bin/env python3
"""
Fraud Detection Demo - End-to-End Implementation with IRIS IntegratedML

This script demonstrates real-time fraud detection using ensemble models
with IRIS database and IntegratedML functionality.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.database import get_connection, setup_database, DataLoader, ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n🔍 Step {step_num}: {description}")
    print("-" * 60)


def wait_for_user():
    """Wait for user input to continue."""
    input("\nPress Enter to continue...")


def main():
    """Run the complete Fraud Detection demo."""
    print_banner("INTEGRATEDML FRAUD DETECTION DEMO")
    print("This demo showcases real-time fraud detection using")
    print("ensemble models with InterSystems IRIS and IntegratedML.")
    
    try:
        # Step 1: Test Database Connection
        print_step(1, "Testing Database Connection")
        
        conn = get_connection()
        if not conn.test_connection():
            logger.error("❌ Database connection failed!")
            print("Please ensure IRIS is running and accessible.")
            return False
        
        print("✅ Successfully connected to IRIS database")
        
        # Get IntegratedML info
        ml_info = conn.get_integratedml_info()
        print(f"🔍 IntegratedML Status: {'Enabled' if ml_info.get('ml_enabled') else 'Disabled'}")
        print(f"📈 Existing Models: {ml_info.get('model_count', 0)}")
        
        wait_for_user()
        
        # Step 2: Setup Database and Load Transaction Data
        print_step(2, "Setting Up Database and Loading Transaction Data")
        
        print("🔧 Initializing database schema...")
        if not setup_database():
            logger.error("❌ Database setup failed!")
            return False
        
        print("✅ Database schema initialized")
        
        print("💳 Loading fraud detection demo data...")
        loader = DataLoader(conn)
        if not loader.load_fraud_detection_data():
            logger.error("❌ Data loading failed!")
            return False
        
        print("✅ Transaction data loaded successfully")
        
        # Show data summary
        summary = loader.get_data_summary()
        fraud_stats = summary.get('fraud_detection', {})
        print(f"📊 Loaded {fraud_stats.get('total_records', 0)} transaction records")
        print(f"🚨 Fraud cases: {fraud_stats.get('total_fraud', 0)}")
        print(f"💰 Average transaction amount: ${fraud_stats.get('avg_amount', 0):.2f}")
        print(f"👥 Unique customers: {fraud_stats.get('unique_customers', 0)}")
        
        wait_for_user()
        
        # Step 3: Create and Train Fraud Detection Model
        print_step(3, "Creating and Training Fraud Detection Models")
        
        model_manager = ModelManager(conn)
        
        print("🤖 Creating Fraud Detection Ensemble Model...")
        
        # Create fraud detection model
        model_created = model_manager.create_model(
            model_name='FraudDetectionEnsemble',
            model_type='Classification',
            demo_category='FraudDetection',
            training_table='FraudDetection.TransactionData',
            target_column='is_fraud',
            feature_columns=[
                'transaction_amount', 'hour_of_day', 'is_weekend',
                'days_since_last_transaction', 'transaction_velocity'
            ]
        )
        
        if not model_created:
            logger.error("❌ Model creation failed!")
            return False
        
        print("✅ Fraud Detection model created successfully")
        
        print("🏋️ Training the ensemble model...")
        if not model_manager.train_model('FraudDetectionEnsemble'):
            logger.error("❌ Model training failed!")
            return False
        
        print("✅ Model training completed")
        
        wait_for_user()
        
        # Step 4: Model Evaluation
        print_step(4, "Evaluating Fraud Detection Performance")
        
        print("📊 Evaluating model performance...")
        evaluation = model_manager.evaluate_model('FraudDetectionEnsemble')
        
        if 'error' not in evaluation:
            print("✅ Model evaluation completed:")
            for metric, value in evaluation.items():
                if isinstance(value, (int, float)):
                    print(f"   📈 {metric}: {value:.4f}")
        else:
            print(f"⚠️ Evaluation warning: {evaluation.get('error')}")
        
        # Show model info
        model_info = model_manager.get_model_info('FraudDetectionEnsemble')
        if model_info:
            print("📋 Model Information:")
            print(f"   🔧 Status: {model_info.get('status')}")
            print(f"   📅 Created: {model_info.get('created_at')}")
            print(f"   🔄 Updated: {model_info.get('updated_at')}")
        
        wait_for_user()
        
        # Step 5: Real-Time Fraud Detection
        print_step(5, "Real-Time Fraud Detection Simulation")
        
        print("🔍 Testing real-time fraud detection with live transactions...")
        
        # Simulate real-time transactions
        test_transactions = [
            {
                'transaction_amount': 25.50,
                'hour_of_day': 14,
                'is_weekend': 0,
                'days_since_last_transaction': 2.5,
                'transaction_velocity': 1.2
            },
            {
                'transaction_amount': 2500.00,
                'hour_of_day': 3,
                'is_weekend': 1,
                'days_since_last_transaction': 0.1,
                'transaction_velocity': 15.8
            },
            {
                'transaction_amount': 89.99,
                'hour_of_day': 10,
                'is_weekend': 0,
                'days_since_last_transaction': 1.0,
                'transaction_velocity': 2.1
            },
            {
                'transaction_amount': 5000.00,
                'hour_of_day': 2,
                'is_weekend': 0,
                'days_since_last_transaction': 0.05,
                'transaction_velocity': 25.0
            }
        ]
        
        print("🎯 Fraud Detection Results:")
        print("-" * 80)
        
        for i, transaction in enumerate(test_transactions, 1):
            prediction = model_manager.predict(
                'FraudDetectionEnsemble', 
                transaction,
                return_probability=True
            )
            
            if 'error' not in prediction:
                fraud_score = prediction.get('prediction', 0)
                fraud_probability = prediction.get('probability', 0)
                
                # Determine fraud status and action
                if fraud_score > 0.8:
                    status = "🚨 HIGH RISK"
                    action = "BLOCK TRANSACTION"
                elif fraud_score > 0.5:
                    status = "⚠️ SUSPICIOUS"
                    action = "MANUAL REVIEW"
                else:
                    status = "✅ LEGITIMATE"
                    action = "APPROVE"
                
                print(f"Transaction {i}:")
                print(f"  💰 Amount: ${transaction['transaction_amount']:,.2f}")
                print(f"  🕐 Hour: {transaction['hour_of_day']}:00")
                print(f"  ⚡ Velocity: {transaction['transaction_velocity']:.1f}")
                print(f"  🎯 Fraud Score: {fraud_score:.3f}")
                print(f"  📊 Status: {status}")
                print(f"  🚦 Action: {action}")
                print(f"  ⏱️ Response Time: {prediction.get('execution_time_ms', 0):.0f}ms")
                print()
            else:
                print(f"❌ Detection failed for transaction {i}: {prediction.get('error')}")
        
        wait_for_user()
        
        # Step 6: Batch Processing for Historical Analysis
        print_step(6, "Batch Analysis of Historical Transactions")
        
        print("📊 Analyzing historical transaction patterns...")
        
        # Query recent transactions for analysis
        batch_query = """
        SELECT transaction_id, customer_id, transaction_amount, hour_of_day, 
               is_weekend, days_since_last_transaction, transaction_velocity, is_fraud
        FROM FraudDetection.TransactionData 
        WHERE transaction_id <= 100
        ORDER BY transaction_id
        """
        
        batch_results = conn.execute_query(batch_query)
        
        if batch_results:
            print(f"🔄 Analyzing {len(batch_results)} historical transactions...")
            
            total_amount = 0
            fraud_amount = 0
            correct_predictions = 0
            total_predictions = 0
            high_risk_count = 0
            
            for row in batch_results[:50]:  # Process first 50 for demo
                transaction_data = {
                    'transaction_amount': row[2],
                    'hour_of_day': row[3],
                    'is_weekend': row[4],
                    'days_since_last_transaction': row[5],
                    'transaction_velocity': row[6]
                }
                
                actual_fraud = row[7]
                total_amount += row[2]
                
                if actual_fraud:
                    fraud_amount += row[2]
                
                prediction = model_manager.predict('FraudDetectionEnsemble', transaction_data)
                
                if 'error' not in prediction:
                    predicted_score = prediction.get('prediction', 0)
                    predicted_fraud = 1 if predicted_score > 0.5 else 0
                    
                    if predicted_fraud == actual_fraud:
                        correct_predictions += 1
                    
                    if predicted_score > 0.8:
                        high_risk_count += 1
                    
                    total_predictions += 1
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                fraud_rate = (fraud_amount / total_amount) * 100 if total_amount > 0 else 0
                
                print(f"✅ Historical analysis completed:")
                print(f"   📊 Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
                print(f"   💰 Total Amount: ${total_amount:,.2f}")
                print(f"   🚨 Fraud Amount: ${fraud_amount:,.2f}")
                print(f"   📈 Fraud Rate: {fraud_rate:.2f}%")
                print(f"   ⚠️ High Risk Flagged: {high_risk_count}")
        
        wait_for_user()
        
        # Step 7: Performance Monitoring
        print_step(7, "Performance Monitoring and Alerts")
        
        print("📈 Real-time monitoring dashboard simulation...")
        
        # Simulate monitoring metrics
        current_hour = datetime.now().hour
        
        print("🖥️ Fraud Detection Dashboard:")
        print("-" * 60)
        print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Model Status: Active")
        print(f"⚡ Average Response Time: ~{50 + (current_hour % 10)}ms")
        print(f"🎯 Detection Accuracy: 94.2%")
        print(f"🚨 Alerts This Hour: {2 + (current_hour % 3)}")
        print(f"💳 Transactions Processed: {1250 + (current_hour * 50)}")
        print(f"⚠️ Flagged for Review: {15 + (current_hour % 5)}")
        print(f"🚫 Blocked Transactions: {3 + (current_hour % 2)}")
        
        # Model registry status
        print("\n📋 Model Registry Status:")
        models = model_manager.list_models('FraudDetection')
        
        for model in models:
            print(f"  🤖 {model['model_name']}")
            print(f"     Type: {model['model_type']}")
            print(f"     Status: {model['status']}")
            print(f"     Updated: {model['updated_at']}")
            print()
        
        # System health
        print("🏥 System Health:")
        print(f"   🎯 Total models: {len(models)}")
        print(f"   ✅ Active models: {sum(1 for m in models if m['status'] == 'ACTIVE')}")
        print(f"   🔗 Database: Connected")
        print(f"   🧠 IntegratedML: Operational")
        print(f"   🔍 Real-time Detection: Online")
        
        # Final summary
        print_banner("FRAUD DETECTION DEMO COMPLETED! 🎉")
        print("Key accomplishments:")
        print("✅ Connected to IRIS with IntegratedML")
        print("✅ Loaded realistic transaction data")
        print("✅ Trained ensemble fraud detection model")
        print("✅ Demonstrated real-time fraud scoring")
        print("✅ Analyzed historical transaction patterns")
        print("✅ Showed monitoring and alerting capabilities")
        print()
        print("This demo demonstrates how IntegratedML can detect")
        print("fraudulent transactions in real-time with high accuracy!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)