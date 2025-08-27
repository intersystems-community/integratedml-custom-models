# DNA Similarity and Classification Demo

## IntegratedML Pluggable Models Framework Solution

This demo recreates the functionality of the [original DNA similarity project](https://openexchange.intersystems.com/package/DNA-similarity-and-classify) while demonstrating how the **IntegratedML Pluggable Models Framework** solves the three main architectural challenges identified in the original implementation.

## 🎯 Original Project Challenges Addressed

### Challenge #1: Hardcoded Vectorization
**Original Problem:**
```python
# Hardcoded in original project
model = SentenceTransformer('stsb-roberta-base-v2')
embeddings = model.encode(human_df['K_mers_str'].tolist(), normalize_embeddings=True)
```

**IntegratedML Solution:**
```yaml
# Configurable via YAML
preprocessing:
  vectorization_strategy: "count_vectorizer"  # Options: count_vectorizer, tfidf_vectorizer, sentence_transformer
  transformer_model: "stsb-roberta-base-v2"   # Configurable model when using transformers
  k_mer_size: 6
  ngram_range: [4, 4]
```

### Challenge #2: Hardcoded Algorithm Selection
**Original Problem:**
```python
# Fixed algorithm in original project
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
```

**IntegratedML Solution:**
```yaml
# Pluggable algorithms via configuration
algorithm: "multinomial_nb"  # Options: multinomial_nb, random_forest, svm, logistic_regression
algorithm_params:
  alpha: 0.1
```

### Challenge #3: Mixed Database Logic
**Original Problem:**
```python
# Mixed business logic and database operations in original
query = """INSERT INTO dc_data.HumanDNA (sequence, kMers, kMersVector, dnaClass) VALUES (?, ?, TO_VECTOR(?), ?)"""
stmt = iris.sql.prepare(query)
for index, row in human_df.iterrows():
    rs = stmt.execute(row['sequence'], row['K_mers_str'], str(row['sequence_vectorized']), row['class'])
```

**IntegratedML Solution:**
```python
# Clean separation with database abstraction
from shared.database.model_manager import ModelManager
model_manager = ModelManager(config['database'])
model_manager.deploy_model(classifier)  # Automated deployment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- InterSystems IRIS with IntegratedML
- Required packages: `pandas`, `scikit-learn`, `sentence-transformers`, `pyyaml`

### Installation
```bash
# Install dependencies
pip install pandas scikit-learn sentence-transformers pyyaml numpy joblib

# Clone the repository
git clone <repository-url>
cd pluggable_iml/demos/dna_similarity
```

### Running the Demo
```bash
# Run the complete demonstration
python dna_demo.py
```

This will demonstrate:
1. **Multiple Vectorization Strategies** - Count vectorizer, TF-IDF, and SentenceTransformer
2. **Multiple Algorithm Options** - MultinomialNB, RandomForest, SVM, LogisticRegression
3. **Clean IRIS Integration** - Database abstraction and automated deployment
4. **Vector Similarity Search** - DNA sequence similarity using embeddings

## 📁 Project Structure

```
demos/dna_similarity/
├── models/
│   └── dna_classifier.py          # Main DNA classification model
├── config/
│   └── dna_model_config.yaml      # Configuration-driven model setup
├── sql/
│   └── create_tables.sql          # Database schema setup
├── data/                          # Sample data directory
├── dna_demo.py                    # Complete demonstration script
└── README.md                      # This documentation
```

## 🧬 How It Works

### 1. DNA Sequence Processing
The framework provides flexible k-mer generation and vectorization:

```python
class DNASequenceClassifier(ClassificationModel):
    def generate_kmers(self, sequence: str, size: int = 6) -> List[str]:
        """Generate k-mers with configurable size"""
        sequence = sequence.upper()
        kmers = [sequence[i:i+size] for i in range(len(sequence) - size + 1)]
        return [kmer for kmer in kmers if len(kmer) == size and 'N' not in kmer]
    
    def _configure_vectorizer(self) -> Any:
        """Configure vectorization strategy from config"""
        strategy = self.vectorization_strategy.lower()
        if strategy == 'count_vectorizer':
            return CountVectorizer(ngram_range=self.ngram_range)
        elif strategy == 'sentence_transformer':
            return SentenceTransformer(self.config.get('transformer_model'))
        # ... more strategies
```

### 2. Configurable Algorithm Selection
Multiple ML algorithms with hyperparameter tuning:

```python
def get_algorithm_instance(self) -> Any:
    """Get configured algorithm from YAML config"""
    algorithm = self.config.get('algorithm', 'multinomial_nb')
    params = self.config.get('algorithm_params', {})
    
    algorithms = {
        'multinomial_nb': MultinomialNB,
        'random_forest': RandomForestClassifier,
        'svm': SVC,
        'logistic_regression': LogisticRegression
    }
    return algorithms[algorithm](**params)
```

### 3. Clean Database Integration
Automated IRIS database operations:

```python
# Automated model deployment
model_manager.deploy_model(classifier)

# Clean similarity search
def find_similar_sequences(query_sequence: str, top_k: int = 5):
    query_embedding = self.encode_sequence(query_sequence)
    return database.vector_search(query_embedding, top_k)
```

## 🔧 Configuration Options

### Vectorization Strategies
```yaml
preprocessing:
  vectorization_strategy: "count_vectorizer"
  # Options:
  # - count_vectorizer: Frequency-based k-mer counting
  # - tfidf_vectorizer: TF-IDF weighted k-mer features  
  # - sentence_transformer: Neural embeddings via transformers
```

### Algorithm Selection
```yaml
algorithm: "multinomial_nb"
# Options:
# - multinomial_nb: Naive Bayes (original algorithm)
# - random_forest: Random Forest ensemble
# - svm: Support Vector Machine
# - logistic_regression: Logistic Regression

algorithm_params:
  alpha: 0.1  # Algorithm-specific parameters
```

### Database Configuration
```yaml
database:
  connection_string: "iris://localhost:1972/USER"
  table_name: "dna_sequences"
  auto_deploy: true
  vector_search:
    enabled: true
    similarity_function: "VECTOR_DOT_PRODUCT"
```

## 📊 Performance Comparison

| Metric | Original Project | IntegratedML Framework |
|--------|-----------------|----------------------|
| **Vectorization** | Hardcoded SentenceTransformer | 3+ configurable strategies |
| **Algorithms** | Fixed MultinomialNB | 4+ pluggable algorithms |
| **Database** | Mixed business/data logic | Clean abstraction layer |
| **Configuration** | Code changes required | YAML-driven configuration |
| **Deployment** | Manual SQL execution | Automated deployment |
| **Monitoring** | No built-in monitoring | Comprehensive logging/metrics |
| **Testing** | Limited test coverage | Full test suite with framework |

## 🎯 Key Benefits

### 1. **Configuration-Driven Development**
- No code changes needed to try different algorithms
- Easy A/B testing of vectorization strategies
- Production-ready configuration management

### 2. **Clean Architecture**
- Separation of concerns (data, models, database)
- Modular and extensible design
- Professional error handling and logging

### 3. **Enterprise-Ready Features**
- Automated model deployment to IRIS
- Version control and model management
- Performance monitoring and metrics
- SQL generation and optimization

### 4. **Developer Productivity**
- Reduced boilerplate code
- Consistent patterns across projects
- Built-in best practices

## 🧪 Example Usage

### Basic Classification
```python
from demos.dna_similarity.models.dna_classifier import DNASequenceClassifier

# Load configuration
config = yaml.safe_load(open('config/dna_model_config.yaml'))

# Initialize classifier
classifier = DNASequenceClassifier(config=config['preprocessing'])

# Classify a sequence
sequence = "ATGAACTGTCCAGCCCCTGTGGAGATCTCCTATGAGAACATGCGTTTTCTGATATCTCACAACCCT"
result = classifier.predict_single_sequence(sequence)

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Similarity Search
```python
from demos.dna_similarity.models.dna_classifier import DNASimilaritySearch

# Initialize similarity search
similarity_search = DNASimilaritySearch(config['preprocessing'])

# Find similar sequences
similar = similarity_search.find_similar_sequences(
    query_sequence=sequence,
    reference_sequences=reference_data,
    top_k=5
)

for seq in similar:
    print(f"Class: {seq['class']}, Similarity: {seq['similarity_score']:.4f}")
```

## 🔬 Scientific Background

### DNA Classification Classes
The demo supports classification into 7 functional gene families:

1. **G protein coupled receptors** - Cell membrane signaling proteins
2. **tyrosine kinase** - Phosphorylation enzymes
3. **tyrosine phosphatase** - Dephosphorylation enzymes  
4. **synthetase** - ATP-dependent synthesis enzymes
5. **synthase** - ATP-independent synthesis enzymes
6. **ion channel** - Membrane transport proteins
7. **transcription factor** - Gene expression regulators

### K-mer Analysis
K-mers are subsequences of length k used in bioinformatics for sequence analysis:
- **K-mer size**: Typically 6 (hexamers) for DNA classification
- **Sliding window**: Overlapping k-mers capture local sequence patterns
- **Vectorization**: K-mers converted to numerical features for ML algorithms

## 📈 Extending the Framework

### Adding New Vectorization Strategies
```python
def _configure_vectorizer(self) -> Any:
    strategy = self.vectorization_strategy.lower()
    if strategy == 'your_new_strategy':
        return YourCustomVectorizer(**self.config)
    # ... existing strategies
```

### Adding New Algorithms
```python
def get_algorithm_instance(self) -> Any:
    algorithms = {
        'your_algorithm': YourCustomAlgorithm,
        # ... existing algorithms
    }
    return algorithms[algorithm](**params)
```

### Custom Preprocessing
```python
class CustomDNAClassifier(DNASequenceClassifier):
    def preprocess_data(self, X, y=None, is_training=True):
        # Add custom preprocessing steps
        X = super().preprocess_data(X, y, is_training)
        # Your custom logic here
        return X
```

## 🚀 Production Deployment

### 1. Database Setup
```sql
-- Run the database setup
@demos/dna_similarity/sql/create_tables.sql
```

### 2. Model Training
```python
# Train with production data
classifier.fit(production_data)

# Deploy to IRIS
model_manager.deploy_model(classifier)
```

### 3. API Integration
```python
# REST API endpoint
@app.route('/api/dna/classify', methods=['POST'])
def classify_dna():
    sequence = request.json['sequence']
    result = classifier.predict_single_sequence(sequence)
    return jsonify(result)
```

## 🤝 Contributing

To contribute to this demo:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Original DNA similarity project by [Davi Massaru Teixeira Muta](https://github.com/Davi-Massaru/DNA-similarity-and-classify)
- InterSystems IRIS Vector Search capabilities
- scikit-learn and SentenceTransformers libraries
- Human DNA Sequences Dataset from Kaggle

---

**Ready to get started?** Run `python dna_demo.py` to see the IntegratedML framework in action! 🧬✨