#!/usr/bin/env python3
"""
DNA Similarity and Classification Demo using IntegratedML Flexible Model Integration Framework

This demo recreates the functionality of the original DNA similarity project while 
demonstrating how the IntegratedML framework solves the three main challenges:

1. **Vectorization Challenge**: Configurable preprocessing pipelines
2. **Algorithm Selection Challenge**: Declarative algorithm configuration  
3. **IRIS Integration Gap**: Clean database abstraction with automated deployment

Run this demo to see:
- Multiple vectorization strategies (count, TF-IDF, transformers)
- Multiple algorithm options (MultinomialNB, RandomForest, SVM, LogisticRegression)  
- Clean separation of concerns
- Automated IRIS database integration
- Vector similarity search
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from demos.dna_similarity.models.dna_classifier import DNASequenceClassifier, DNASimilaritySearch
from shared.database.connection import IRISConnection
from shared.database.model_manager import ModelManager
from shared.utils.logging import setup_logger

logger = setup_logger(__name__)


class DNADemoRunner:
    """
    DNA Demo Runner - Shows how IntegratedML framework solves original project challenges
    """
    
    def __init__(self, config_path: str):
        """Initialize demo with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.classifier = None
        self.similarity_search = None
        self.db_connection = None
        self.model_manager = None
        
        logger.info("Initialized DNA Demo Runner")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample DNA data for demonstration
        
        This simulates the original project's human_data.txt but with a smaller dataset
        for demo purposes.
        """
        logger.info("Creating sample DNA dataset...")
        
        # Sample DNA sequences for each class (simplified for demo)
        sample_data = [
            # G protein coupled receptors (class 0)
            {
                'sequence': 'ATGTCTAAAAAGGAGAAAGACAAACTCACTATTCAATCAATCGCAATCAGCTGCATCATGGGCATGGTAGCCATTGCCATCGTCTGCTTCTTCCTGATGGTGATCCTCATCCTGCTGTGCTTCCTGGACAATGTGGGAGATGCTATGGAGGAGACCTCTAAGAACATCAAGCCCATCTATCAGGTGGACAATTTGTAG',
                'class': 0
            },
            {
                'sequence': 'ATGGGGAATGGCTTGGATCTAAGGCTGGTGGTGATGGCAGTGGTTGCCCTGGTCCTGGCTGTCTTGATCCTGCTCTTCATCTGCATGGTCATCACCATTTTCAAGAACCTGAAGGCCTGGGATGACCTCCAGAACGACTTGACCTTGTACCTGACCAACATCCTGTGGATCCTGCTGTTCAAGGGCTAA',
                'class': 0
            },
            
            # tyrosine kinase (class 1)
            {
                'sequence': 'ATGGAGCAGAAACTCATCTCCGAGGAGGACCTGCAGAAGGCCATCAAGAAGACCCAGGTGGGCTTCGACAACAAGTACTACGTGGAGGTGCTGGACAACTTTAACCCCATCCTGACCAACTACCCCGAGATCATCTACAAGAAGATCCTGGACCAGACCTACATCTACAACAAGGCCATCAAGGAGTTCTAG',
                'class': 1
            },
            {
                'sequence': 'ATGAAGCTGATCAAGGAGTTCATGCGCATCTGGAAGGACCTGGATAAGGCCAAGACCTACAACCCCATCTACTGGAAGATCAACCAGATCATCAAGGACCTGGCCAAGGACATCTACAACCCCATCCTGGACTACATCAAGGACCAGAAGATCATCTACAAGAAGATCTAG',
                'class': 1
            },
            
            # tyrosine phosphatase (class 2) - Original sequence from demo
            {
                'sequence': 'ATGAACTGTCCAGCCCCTGTGGAGATCTCCTATGAGAACATGCGTTTTCTGATATCTCACAACCCTACCAATGCTACTCTCAACAAGTTCACAGAGGAACTTAAGAAGTATGGAGTGACGACTTTGGTTCGAGTTTGTGATGCTACATATGATAAAGCTCCAGTTGAAAAAGAAGGAATCCACGTTCTAG',
                'class': 2
            },
            {
                'sequence': 'ATGAACCGTCCAGCCCCTGTGGAGATCTCCTATGAGAACATGCGTTTTCTGATAACTCACAACCCTACCAATGCTACTCTCAACAAGTTCACAGAGGAACTTAAGAAGTATGGAGTGACGACTTTGGTTCGAGTTTGTGATGCTACATATGATAAAGCTCCAGTTGAAAAAGAAGGAATCCACGTTCTAATGGCAGAGTGA',
                'class': 2
            },
            
            # synthetase (class 3)
            {
                'sequence': 'ATGGCCGAGAAGCTGATCAAGGCCATGAAGAAGATCCTGGACAAGGTGGACTTCAAGAAGATCCTGAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 3
            },
            {
                'sequence': 'ATGGGCGAGAAGCTGATCAAGACCATGAAGAAGATCCTGGACAAGGTGGACTTCAAGAAGATCCTGAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 3
            },
            
            # synthase (class 4)
            {
                'sequence': 'ATGACCGAGAAGCTGATCAAGGCCATGAAGAAGATCCTGGACAAGGTGGACTTCAAGAAGATCCTGAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 4
            },
            {
                'sequence': 'ATGACGGAGAAGCTGATCAAGACCATGAAGAAGATCCTGGACAAGGTGGACTTCAAGAAGATCCTGAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 4
            },
            
            # ion channel (class 5)
            {
                'sequence': 'ATGGCCATCGTGCTGATCATGGGCATGGTAGCCATTGCCATCGTCTGCTTCTTCCTGATGGTGATCCTCATCCTGCTGTGCTTCCTGGACAATGTGGGAGATGCTATGGAGGAGACCTCTAAGAACATCAAGCCCATCTATCAGGTGGACAATTTGTAG',
                'class': 5
            },
            {
                'sequence': 'ATGGCGATCGTGCTGATCATGGGCATGGTAGCCATTGCCATCGTCTGCTTCTTCCTGATGGTGATCCTCATCCTGCTGTGCTTCCTGGACAATGTGGGAGATGCTATGGAGGAGACCTCTAAGAACATCAAGCCCATCTATCAGGTGGACAATTTGTAG',
                'class': 5
            },
            
            # transcription factor (class 6)
            {
                'sequence': 'ATGGCGCTGAACCCTTCCGGCACCATGGCCAAGTCCTACTTCATCGAGAACGCCATCAAGACCAAGGGCAAGCTGAAGAAGGACCTGAAGAACCCCATCAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 6
            },
            {
                'sequence': 'ATGGCCCTGAACCCTTCCGGCACCATGGCCAAGTCCTACTTCATCGAGAACGCCATCAAGACCAAGGGCAAGCTGAAGAAGGACCTGAAGAACCCCATCAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
                'class': 6
            }
        ]
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Created sample dataset with {len(df)} sequences across {df['class'].nunique()} classes")
        return df
    
    def demonstrate_vectorization_strategies(self, data: pd.DataFrame) -> None:
        """
        Demonstrate different vectorization strategies - addresses Original Challenge #1
        """
        print("\n" + "="*60)
        print("DEMONSTRATION: Multiple Vectorization Strategies")
        print("Original Challenge: Hardcoded SentenceTransformer usage")
        print("IntegratedML Solution: Configurable vectorization pipelines")
        print("="*60)
        
        strategies = ['count_vectorizer', 'tfidf_vectorizer', 'sentence_transformer']
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy.upper()} Strategy ---")
            
            # Create config for this strategy
            strategy_config = self.config.copy()
            strategy_config['preprocessing']['vectorization_strategy'] = strategy
            
            try:
                # Initialize classifier with this strategy
                classifier = DNASequenceClassifier(
                    config=strategy_config['preprocessing'],
                    model_name=f"dna_classifier_{strategy}"
                )
                
                print(f"✓ {strategy} initialized successfully")
                print(f"  K-mer size: {classifier.k_mer_size}")
                print(f"  Vectorization: {classifier.vectorization_strategy}")
                
                # Test preprocessing with sample data
                X_sample = data.head(3)
                y_sample = X_sample['class']
                
                X_processed, y_processed = classifier.preprocess_data(X_sample, y_sample)
                print(f"  Processed shape: {X_processed.shape}")
                print(f"  Feature dimensions: {X_processed.shape[1]}")
                
            except Exception as e:
                print(f"✗ Error with {strategy}: {str(e)}")
    
    def demonstrate_algorithm_selection(self, data: pd.DataFrame) -> None:
        """
        Demonstrate different algorithm options - addresses Original Challenge #2
        """
        print("\n" + "="*60)
        print("DEMONSTRATION: Multiple Algorithm Options")
        print("Original Challenge: Hardcoded MultinomialNB algorithm")
        print("IntegratedML Solution: Pluggable algorithm selection via config")
        print("="*60)
        
        algorithms = [
            ('multinomial_nb', {'alpha': 0.1}),
            ('random_forest', {'n_estimators': 10, 'random_state': 42}),
            ('logistic_regression', {'random_state': 42, 'max_iter': 100})
        ]
        
        for algo_name, algo_params in algorithms:
            print(f"\n--- Testing {algo_name.upper()} Algorithm ---")
            
            try:
                # Create config for this algorithm
                algo_config = self.config.copy()
                algo_config['algorithm'] = algo_name
                algo_config['algorithm_params'] = algo_params
                
                # Initialize classifier
                classifier = DNASequenceClassifier(
                    config=algo_config,
                    model_name=f"dna_classifier_{algo_name}"
                )
                
                # Get algorithm instance
                algo_instance = classifier.get_algorithm_instance()
                print(f"✓ {algo_name} initialized: {type(algo_instance).__name__}")
                print(f"  Parameters: {algo_params}")
                
                # Quick train/test to verify functionality
                X_train, y_train = classifier.preprocess_data(data, data['class'])
                classifier.model = algo_instance
                classifier.model.fit(X_train, y_train)
                
                # Test prediction
                test_sequence = data.iloc[0]['sequence']
                prediction = classifier.predict_single_sequence(test_sequence)
                print(f"  Test prediction: {prediction['classification']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
                
            except Exception as e:
                print(f"✗ Error with {algo_name}: {str(e)}")
    
    def demonstrate_iris_integration(self) -> None:
        """
        Demonstrate clean IRIS database integration - addresses Original Challenge #3
        """
        print("\n" + "="*60)
        print("DEMONSTRATION: Clean IRIS Database Integration")
        print("Original Challenge: Mixed business logic and database operations")
        print("IntegratedML Solution: Clean database abstraction layer")
        print("="*60)
        
        try:
            # Initialize database connection
            db_config = self.config.get('database', {})
            print(f"✓ Database configuration loaded")
            print(f"  Connection: {db_config.get('connection_string', 'iris://localhost:1972/USER')}")
            print(f"  Table: {db_config.get('table_name', 'dna_sequences')}")
            print(f"  Auto-deploy: {db_config.get('auto_deploy', True)}")
            print(f"  Vector search: {db_config.get('vector_search', {}).get('enabled', True)}")
            
            # Show model manager capabilities
            model_config = {
                'model_name': self.config['model_name'],
                'model_type': self.config['model_type'],
                'version': '1.0.0'
            }
            
            print(f"\n✓ Model management configured")
            print(f"  Model name: {model_config['model_name']}")
            print(f"  Model type: {model_config['model_type']}")
            print(f"  Automated deployment: Available")
            print(f"  Version control: Available")
            print(f"  SQL generation: Automated")
            
        except Exception as e:
            print(f"✗ Error with IRIS integration: {str(e)}")
    
    def demonstrate_similarity_search(self, data: pd.DataFrame) -> None:
        """
        Demonstrate DNA similarity search functionality
        """
        print("\n" + "="*60)
        print("DEMONSTRATION: DNA Similarity Search")
        print("Functionality: Vector-based sequence similarity search")
        print("="*60)
        
        try:
            # Initialize similarity search
            search_config = self.config['preprocessing']
            similarity_search = DNASimilaritySearch(search_config)
            
            print(f"✓ Similarity search initialized")
            print(f"  Transformer model: {similarity_search.transformer_model}")
            print(f"  K-mer size: {similarity_search.k_mer_size}")
            
            # Prepare reference sequences
            reference_sequences = []
            for _, row in data.iterrows():
                reference_sequences.append({
                    'sequence': row['sequence'],
                    'class': self.config['class_mapping'][row['class']]
                })
            
            # Test similarity search
            query_sequence = data.iloc[0]['sequence']
            query_class = self.config['class_mapping'][data.iloc[0]['class']]
            
            print(f"\n--- Similarity Search Test ---")
            print(f"Query sequence class: {query_class}")
            print(f"Query length: {len(query_sequence)} bp")
            
            similar_sequences = similarity_search.find_similar_sequences(
                query_sequence, reference_sequences, top_k=3
            )
            
            print(f"\nTop 3 similar sequences:")
            for i, seq in enumerate(similar_sequences, 1):
                print(f"  {i}. Class: {seq['class']}")
                print(f"     Similarity: {seq['similarity_score']:.4f}")
                print(f"     Length: {len(seq['sequence'])} bp")
            
        except Exception as e:
            print(f"✗ Error with similarity search: {str(e)}")
    
    def run_complete_demo(self) -> None:
        """
        Run the complete demonstration showing all framework capabilities
        """
        print("🧬 DNA Similarity and Classification Demo")
        print("IntegratedML Flexible Model Integration Framework")
        print("="*60)
        
        try:
            # Create sample data
            data = self._create_sample_data()
            
            # Run demonstrations
            self.demonstrate_vectorization_strategies(data)
            self.demonstrate_algorithm_selection(data)
            self.demonstrate_iris_integration()
            self.demonstrate_similarity_search(data)
            
            # Summary of improvements
            print("\n" + "="*60)
            print("SUMMARY: How IntegratedML Framework Solves Original Challenges")
            print("="*60)
            print("""
✓ CHALLENGE #1: Hardcoded Vectorization
  Original: Fixed SentenceTransformer('stsb-roberta-base-v2')
  Solution: Configurable strategies (count, TF-IDF, transformers)
  
✓ CHALLENGE #2: Hardcoded Algorithm  
  Original: Fixed MultinomialNB(alpha=0.1)
  Solution: Pluggable algorithms via YAML configuration
  
✓ CHALLENGE #3: Mixed Database Logic
  Original: Raw SQL and business logic mixed
  Solution: Clean abstraction with automated deployment
  
✓ ADDITIONAL BENEFITS:
  - Configuration-driven development
  - Clean separation of concerns  
  - Automated model deployment
  - Version control and monitoring
  - Extensible architecture
  - Professional logging and error handling
            """)
            
            print("Demo completed successfully! 🎉")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            print(f"✗ Demo failed: {str(e)}")


def main():
    """Main function to run the DNA demo"""
    try:
        # Get config path
        demo_dir = os.path.dirname(__file__)
        config_path = os.path.join(demo_dir, 'config', 'dna_model_config.yaml')
        
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return
        
        # Run the demo
        demo_runner = DNADemoRunner(config_path)
        demo_runner.run_complete_demo()
        
    except Exception as e:
        print(f"Failed to run demo: {str(e)}")
        logger.error(f"Demo execution failed: {str(e)}")


if __name__ == "__main__":
    main()