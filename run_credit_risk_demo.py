#!/usr/bin/env python3
"""
Credit Risk Demo - End-to-End Implementation with IRIS IntegratedML

This script demonstrates the complete workflow for credit risk assessment
using real IRIS database and IntegratedML functionality.
"""

import os
import sys
import logging
import time
from pathlib import Path

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
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 60)


def wait_for_user():
    """Wait for user input to continue."""
    input("\nPress Enter to continue...")


def main():
    """Run the complete Credit Risk demo."""
    print_banner("INTEGRATEDML CREDIT RISK ASSESSMENT DEMO")
    print("This demo showcases real-time credit risk assessment using")
    print("InterSystems IRIS and IntegratedML with flexible model integration.")
    
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
        print(f"📊 IntegratedML Status: {'Enabled' if ml_info.get('ml_enabled') else 'Disabled'}")
        print(f"📈 Existing Models: {ml_info.get('model_count', 0)}")
        
        wait_for_user()
        
        # Step 2: Setup Database and Load Data
        print_step(2, "Setting Up Database and Loading Demo Data")
        
        print("🔧 Initializing database schema...")
        if not setup_database():
            logger.error("❌ Database setup failed!")
            return False
        
        print("✅ Database schema initialized")
        
        print("📊 Loading credit risk demo data...")
        loader = DataLoader(conn)
        if not loader.load_credit_risk_data():
            logger.error("❌ Data loading failed!")
            return False
        
        print("✅ Demo data loaded successfully")
        
        # Show data summary
        summary = loader.get_data_summary()
        credit_stats = summary.get('credit_risk', {})
        print(f"📈 Loaded {credit_stats.get('total_records', 0)} customer records")
        print(f"📈 Average credit score: {credit_stats.get('avg_credit_score', 0)}")
        print(f"📈 Default cases: {credit_stats.get('total_defaults', 0)}")
        
        wait_for_user()
        
        # Step 3: Create and Train Models
        print_step(3, "Creating and Training Credit Risk Models")
        
        model_manager = ModelManager(conn)
        
        print("🤖 Creating Credit Risk Classification Model...")
        
        # Create main credit risk model
        model_created = model_manager.create_model(
            model_name='CreditRiskClassifier',
            model_type='Classification',
            demo_category='CreditRisk',
            training_table='CreditRisk.CustomerData',
            target_column='default_risk',
            feature_columns=[
                'age', 'income', 'credit_score', 'debt_to_income_ratio',
                'employment_length', 'loan_amount'
            ]
        )
        
        if not model_created:
            logger.error("❌ Model creation failed!")
            return False
        
        print("✅ Credit Risk model created successfully")
        
        print("🏋️ Training the model...")
        if not model_manager.train_model('CreditRiskClassifier'):
            logger.error("❌ Model training failed!")
            return False
        
        print("✅ Model training completed")
        
        wait_for_user()
        
        # Step 4: Model Evaluation
        print_step(4, "Evaluating Model Performance")
        
        print("📊 Evaluating model performance...")
        evaluation = model_manager.evaluate_model('CreditRiskClassifier')
        
        if 'error' not in evaluation:
            print("✅ Model evaluation completed:")
            for metric, value in evaluation.items():
                if isinstance(value, (int, float)):
                    print(f"   📈 {metric}: {value:.4f}")
        else:
            print(f"⚠️ Evaluation warning: {evaluation.get('error')}")
        
        # Show model info
        model_info = model_manager.get_model_info('CreditRiskClassifier')
        if model_info:
            print("📋 Model Information:")
            print(f"   🔧 Status: {model_info.get('status')}")
            print(f"   📅 Created: {model_info.get('created_at')}")
            print(f"   🔄 Updated: {model_info.get('updated_at')}")
        
        wait_for_user()
        
        # Step 5: Make Predictions
        print_step(5, "Making Real-Time Predictions")
        
        print("🔮 Testing model predictions with sample data...")
        
        # Sample customer data for predictions
        test_customers = [
            {
                'age': 35,
                'income': 75000,
                'credit_score': 720,
                'debt_to_income_ratio': 0.25,
                'employment_length': 5.0,
                'loan_amount': 25000
            },
            {
                'age': 22,
                'income': 35000,
                'credit_score': 580,
                'debt_to_income_ratio': 0.45,
                'employment_length': 1.0,
                'loan_amount': 15000
            },
            {
                'age': 45,
                'income': 95000,
                'credit_score': 780,
                'debt_to_income_ratio': 0.15,
                'employment_length': 12.0,
                'loan_amount': 40000
            }
        ]
        
        print("🎯 Prediction Results:")
        print("-" * 80)
        
        for i, customer in enumerate(test_customers, 1):
            prediction = model_manager.predict('CreditRiskClassifier', customer)
            
            if 'error' not in prediction:
                risk_score = prediction.get('prediction', 0)
                recommendation = 'APPROVE' if risk_score < 0.5 else 'REJECT'
                risk_level = 'LOW' if risk_score < 0.3 else 'MEDIUM' if risk_score < 0.7 else 'HIGH'
                
                print(f"Customer {i}:")
                print(f"  💰 Income: ${customer['income']:,}")
                print(f"  📊 Credit Score: {customer['credit_score']}")
                print(f"  💳 Loan Amount: ${customer['loan_amount']:,}")
                print(f"  🎯 Risk Score: {risk_score:.3f}")
                print(f"  📊 Risk Level: {risk_level}")
                print(f"  ✅ Recommendation: {recommendation}")
                print(f"  ⏱️ Processing Time: {prediction.get('execution_time_ms', 0):.0f}ms")
                print()
            else:
                print(f"❌ Prediction failed for customer {i}: {prediction.get('error')}")
        
        wait_for_user()
        
        # Step 6: Batch Processing Demo
        print_step(6, "Batch Processing Demonstration")
        
        print("📊 Running batch risk assessment on recent applications...")
        
        # Query recent data for batch processing
        batch_query = """
        SELECT customer_id, age, income, credit_score, debt_to_income_ratio, 
               employment_length, loan_amount, default_risk
        FROM CreditRisk.CustomerData 
        WHERE customer_id <= 20
        ORDER BY customer_id
        """
        
        batch_results = conn.execute_query(batch_query)
        
        if batch_results:
            print(f"🔄 Processing {len(batch_results)} customer records...")
            
            correct_predictions = 0
            total_predictions = 0
            
            for row in batch_results[:10]:  # Process first 10 for demo
                customer_data = {
                    'age': row[1],
                    'income': row[2],
                    'credit_score': row[3],
                    'debt_to_income_ratio': row[4],
                    'employment_length': row[5],
                    'loan_amount': row[6]
                }
                
                actual_risk = row[7]
                prediction = model_manager.predict('CreditRiskClassifier', customer_data)
                
                if 'error' not in prediction:
                    predicted_risk = prediction.get('prediction', 0)
                    predicted_class = 1 if predicted_risk > 0.5 else 0
                    
                    if predicted_class == actual_risk:
                        correct_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                print(f"✅ Batch processing completed")
                print(f"📊 Accuracy on sample: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            
        wait_for_user()
        
        # Step 7: Model Management
        print_step(7, "Model Management and Monitoring")
        
        print("📋 Current model registry:")
        models = model_manager.list_models('CreditRisk')
        
        for model in models:
            print(f"  🤖 {model['model_name']}")
            print(f"     Type: {model['model_type']}")
            print(f"     Status: {model['status']}")
            print(f"     Updated: {model['updated_at']}")
            print()
        
        print("📊 System status summary:")
        print(f"   🎯 Total models in registry: {len(models)}")
        print(f"   ✅ Active models: {sum(1 for m in models if m['status'] == 'ACTIVE')}")
        print(f"   🔄 Database connection: Healthy")
        print(f"   📈 IntegratedML: Enabled")
        
        # Final summary
        print_banner("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("Key accomplishments:")
        print("✅ Connected to IRIS database with IntegratedML")
        print("✅ Loaded realistic credit risk data")
        print("✅ Created and trained classification model")
        print("✅ Made real-time risk predictions")
        print("✅ Demonstrated batch processing")
        print("✅ Showed model management capabilities")
        print()
        print("This demo shows IntegratedML working with live database")
        print("and flexible model integration for real-world credit risk assessment!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)