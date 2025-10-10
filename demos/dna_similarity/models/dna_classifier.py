#!/usr/bin/env python3
"""
DNA Sequence Classification Model using IntegratedML Pluggable Models Framework

This demo shows how the IntegratedML framework solves the challenges from the original 
DNA similarity project:

1. **Vectorization Challenge**: Configurable preprocessing pipelines instead of hardcoded transformers
2. **Algorithm Selection Challenge**: Declarative algorithm configuration via YAML
3. **IRIS Integration Gap**: Clean database abstraction with automated SQL deployment

Original Project Issues Addressed:
- Hardcoded SentenceTransformer usage -> Configurable vectorization strategies
- Hardcoded MultinomialNB algorithm -> Pluggable algorithm selection
- Mixed business/data logic -> Clean separation of concerns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sentence_transformers import SentenceTransformer

from shared.models.classification import ClassificationModel
from shared.utils.logging import setup_logger

logger = setup_logger(__name__)


class DNASequenceClassifier(ClassificationModel):
    """
    DNA Sequence Classification Model with Configurable Vectorization and Algorithm Selection
    
    Demonstrates how IntegratedML framework solves the original DNA project's limitations:
    
    1. **Vectorization Flexibility**: Multiple vectorization strategies (k-mer counting, 
       TF-IDF, transformer embeddings) configured via YAML instead of hardcoded
    
    2. **Algorithm Pluggability**: Support for multiple ML algorithms (MultinomialNB, 
       RandomForest, SVM, LogisticRegression) with hyperparameter tuning
    
    3. **Clean Architecture**: Separation of data processing, model training, and deployment
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Extract config from kwargs (may be passed as 'config' parameter)
        self.config = kwargs.get('config', self.parameters)
        self.k_mer_size = self.config.get('k_mer_size', 6)
        self.max_kmers = self.config.get('max_kmers', 5000)
        self.vectorization_strategy = self.config.get('vectorization_strategy', 'count_vectorizer')
        self.ngram_range = tuple(self.config.get('ngram_range', [4, 4]))
        
        # DNA class mapping - addresses original hardcoded mappings
        self.class_mapping = {
            0: 'G protein coupled receptors',
            1: 'tyrosine kinase', 
            2: 'tyrosine phosphatase',
            3: 'synthetase',
            4: 'synthase',
            5: 'ion channel',  # Fixed typo from original 'lon channel'
            6: 'transcription factor'
        }
        
        # Initialize vectorizer and scaler
        self.vectorizer = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        
        logger.info(f"Initialized DNA Classifier with k-mer size: {self.k_mer_size}, "
                   f"vectorization: {self.vectorization_strategy}")
    
    def generate_kmers(self, sequence: str, size: Optional[int] = None, 
                      max_length: Optional[int] = None) -> List[str]:
        """
        Generate k-mers from DNA sequence
        
        Args:
            sequence: DNA sequence string
            size: K-mer size (default from config)
            max_length: Maximum number of k-mers (default from config)
            
        Returns:
            List of k-mer strings
        """
        if size is None:
            size = self.k_mer_size
        if max_length is None:
            max_length = self.max_kmers
            
        # Convert to uppercase for consistency
        sequence = sequence.upper()
        
        # Generate k-mers with sliding window
        kmers = [sequence[i:i+size] for i in range(len(sequence) - size + 1)]
        
        # Filter out invalid k-mers and limit length
        valid_kmers = [kmer for kmer in kmers if len(kmer) == size and 'N' not in kmer]
        
        if len(valid_kmers) == 0:
            return [sequence[:size]] if len(sequence) >= size else [sequence]
            
        return valid_kmers[:max_length]
    
    def _configure_vectorizer(self) -> Any:
        """
        Configure vectorization strategy based on config - solves original hardcoded limitation
        
        Returns:
            Configured vectorizer instance
        """
        strategy = self.vectorization_strategy.lower()
        
        if strategy == 'count_vectorizer':
            return CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.config.get('max_features', 10000),
                min_df=self.config.get('min_df', 1),
                max_df=self.config.get('max_df', 1.0)
            )
        elif strategy == 'tfidf_vectorizer':
            return TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.config.get('max_features', 10000),
                min_df=self.config.get('min_df', 1),
                max_df=self.config.get('max_df', 1.0)
            )
        elif strategy == 'sentence_transformer':
            # Use SentenceTransformer for embeddings like original, but configurable
            model_name = self.config.get('transformer_model', 'stsb-roberta-base-v2')
            return SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unsupported vectorization strategy: {strategy}")
    
    def preprocess_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                       is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess DNA sequences with configurable vectorization
        
        Args:
            X: DataFrame with 'sequence' column
            y: Target labels (for training)
            is_training: Whether this is training data
            
        Returns:
            Tuple of (processed_features, processed_labels)
        """
        # Generate k-mers for all sequences
        logger.info("Generating k-mers from DNA sequences...")
        X_kmers = X['sequence'].apply(lambda seq: ' '.join(self.generate_kmers(seq)))
        
        # Configure and apply vectorization
        if is_training:
            self.vectorizer = self._configure_vectorizer()
            
        if self.vectorization_strategy == 'sentence_transformer':
            # Handle transformer embeddings
            if is_training:
                logger.info("Computing transformer embeddings...")
                embeddings = self.vectorizer.encode(X_kmers.tolist(), normalize_embeddings=True)
                X_processed = embeddings
            else:
                embeddings = self.vectorizer.encode(X_kmers.tolist(), normalize_embeddings=True)
                X_processed = embeddings
        else:
            # Handle count/tfidf vectorization
            if is_training:
                logger.info(f"Fitting {self.vectorization_strategy}...")
                X_processed = self.vectorizer.fit_transform(X_kmers).toarray()
            else:
                X_processed = self.vectorizer.transform(X_kmers).toarray()
        
        # Optional scaling for transformer embeddings
        if self.vectorization_strategy == 'sentence_transformer' and self.config.get('scale_embeddings', False):
            if is_training:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
            else:
                X_processed = self.scaler.transform(X_processed)
        
        # Process labels if provided
        y_processed = None
        if y is not None:
            if is_training:
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = self.label_encoder.transform(y)
        
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        return X_processed, y_processed
    
    def get_algorithm_instance(self) -> Any:
        """
        Get configured algorithm instance - solves original hardcoded algorithm limitation
        
        Returns:
            Configured scikit-learn estimator
        """
        algorithm = self.config.get('algorithm', 'multinomial_nb')
        params = self.config.get('algorithm_params', {})
        
        if algorithm == 'multinomial_nb':
            return MultinomialNB(**params)
        elif algorithm == 'random_forest':
            return RandomForestClassifier(**params)
        elif algorithm == 'svm':
            return SVC(probability=True, **params)  # Enable probability for predict_proba
        elif algorithm == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def predict_single_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Predict classification for a single DNA sequence
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary with classification and probabilities
        """
        # Create DataFrame for consistent preprocessing
        X_single = pd.DataFrame({'sequence': [sequence]})
        
        # Preprocess
        X_processed, _ = self.preprocess_data(X_single, is_training=False)
        
        # Predict
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Format results
        result = {
            'classification': self.class_mapping[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                self.class_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
    
    def save_model_artifacts(self, model_dir: str) -> None:
        """Save model and preprocessing artifacts"""
        super().save_model_artifacts(model_dir)
        
        # Save DNA-specific artifacts
        if self.vectorizer:
            vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
            joblib.dump(self.vectorizer, vectorizer_path)
            
        if self.scaler:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        
        logger.info(f"Saved DNA classifier artifacts to {model_dir}")
    
    def load_model_artifacts(self, model_dir: str) -> None:
        """Load model and preprocessing artifacts"""
        super().load_model_artifacts(model_dir)
        
        # Load DNA-specific artifacts
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
            
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
            
        logger.info(f"Loaded DNA classifier artifacts from {model_dir}")


class DNASimilaritySearch:
    """
    DNA Similarity Search using Vector Embeddings
    
    Demonstrates clean separation of similarity search from classification,
    addressing the original project's mixed concerns.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.transformer_model = self.config.get('transformer_model', 'stsb-roberta-base-v2')
        self.k_mer_size = self.config.get('k_mer_size', 6)
        self.max_kmers = self.config.get('max_kmers', 5000)
        
        # Initialize transformer for embeddings
        self.embedder = SentenceTransformer(self.transformer_model)
        logger.info(f"Initialized DNA similarity search with model: {self.transformer_model}")
    
    def generate_kmers(self, sequence: str) -> List[str]:
        """Generate k-mers from DNA sequence (consistent with classifier)"""
        sequence = sequence.upper()
        kmers = [sequence[i:i+self.k_mer_size] for i in range(len(sequence) - self.k_mer_size + 1)]
        valid_kmers = [kmer for kmer in kmers if len(kmer) == self.k_mer_size and 'N' not in kmer]
        
        if len(valid_kmers) == 0:
            return [sequence[:self.k_mer_size]] if len(sequence) >= self.k_mer_size else [sequence]
            
        return valid_kmers[:self.max_kmers]
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode DNA sequence to vector representation
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Vector embedding
        """
        kmers = self.generate_kmers(sequence)
        kmers_str = ' '.join(kmers)
        embedding = self.embedder.encode(kmers_str, normalize_embeddings=True)
        return embedding
    
    def find_similar_sequences(self, query_sequence: str, reference_sequences: List[Dict[str, Any]], 
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar DNA sequences using vector similarity
        
        Args:
            query_sequence: Query DNA sequence
            reference_sequences: List of reference sequences with metadata
            top_k: Number of top similar sequences to return
            
        Returns:
            List of similar sequences with similarity scores
        """
        # Encode query sequence
        query_embedding = self.encode_sequence(query_sequence)
        
        # Encode all reference sequences and compute similarities
        similarities = []
        for ref_seq in reference_sequences:
            ref_embedding = self.encode_sequence(ref_seq['sequence'])
            
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = np.dot(query_embedding, ref_embedding)
            
            similarities.append({
                'sequence': ref_seq['sequence'],
                'class': ref_seq.get('class', 'unknown'),
                'similarity_score': float(similarity),
                'metadata': ref_seq.get('metadata', {})
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]


if __name__ == "__main__":
    # Demo usage
    config = {
        'algorithm': 'multinomial_nb',
        'vectorization_strategy': 'count_vectorizer',
        'k_mer_size': 6,
        'ngram_range': [4, 4],
        'algorithm_params': {'alpha': 0.1}
    }
    
    classifier = DNASequenceClassifier(config=config, model_name="dna_classifier_demo")
    print("DNA Classifier initialized successfully!")
    print(f"Vectorization strategy: {classifier.vectorization_strategy}")
    print(f"K-mer size: {classifier.k_mer_size}")