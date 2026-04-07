# DNA SIMILARITY DEMO

## OVERVIEW
DNA sequence classification using configurable k-mer vectorization — demonstrates how IntegratedML framework replaces hardcoded SentenceTransformer + MultinomialNB from original OEX project with YAML-driven algorithm selection.

## STRUCTURE
```
dna_similarity/
├── models/dna_classifier.py    # DNASequenceClassifier(ClassificationModel) — 383 lines
├── sql/create_tables.sql       # HumanDNA table with VECTOR column
├── dna_demo.py                 # Self-contained demo script
└── tests/test_integration.py  # Integration test (requires IRIS)
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Main classifier | `models/dna_classifier.py:37` | `DNASequenceClassifier(ClassificationModel)` |
| K-mer generation | `models/dna_classifier.py` | `generate_kmers(seq, size=6)` |
| Vectorization config | constructor `**kwargs` → `self.vectorization_strategy` | `count_vectorizer`, `tfidf_vectorizer`, `sentence_transformer` |
| Algorithm selection | `config.get('algorithm', 'multinomial_nb')` | `multinomial_nb`, `random_forest`, `svm`, `logistic_regression` |
| Vector similarity SQL | `sql/create_tables.sql` | Uses `VECTOR_DOT_PRODUCT` for k-NN search |

## CONVENTIONS
- Config flows: SQL `USING` JSON → `**kwargs` → `self.config = kwargs.get('config', self.parameters)`
- K-mer default: size=6, max_kmers=5000, ngram_range=(4,4)
- 7 classification targets: G protein coupled receptors, tyrosine kinase/phosphatase, synthetase/synthase, ion channel, transcription factor
- `sentence_transformers` dep required for transformer strategy — heavy, optional

## CRITICAL
`DNASequenceClassifier` imports `from shared.models.classification import ClassificationModel` — **that module doesn't exist**. Demo will fail to import until `shared/models/classification.py` is created.

## ANTI-PATTERNS
- Don't hardcode vectorization strategy — use `config` dict pattern
- No conftest.py, no shared fixtures; integration test requires IRIS + `@pytest.mark.iris`
