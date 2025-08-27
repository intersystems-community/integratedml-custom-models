-- DNA Similarity Demo - Database Setup
-- IntegratedML Pluggable Models Framework
-- 
-- This script creates the necessary tables for the DNA similarity demo,
-- demonstrating clean database integration vs the original project's mixed approach

-- Drop existing tables if they exist
DROP TABLE IF EXISTS dna_similarity.sequences;
DROP TABLE IF EXISTS dna_similarity.models;
DROP TABLE IF EXISTS dna_similarity.predictions;

-- Create schema for DNA similarity demo
CREATE SCHEMA IF NOT EXISTS dna_similarity;

-- Create DNA sequences table with vector support
CREATE TABLE dna_similarity.sequences (
    id INTEGER IDENTITY PRIMARY KEY,
    sequence VARCHAR(MAX) NOT NULL,
    sequence_length INTEGER COMPUTED (LEN(sequence)),
    k_mers VARCHAR(MAX),
    sequence_vector VECTOR(DOUBLE, 768),  -- SentenceTransformer embedding dimension
    dna_class INTEGER NOT NULL,
    dna_class_name VARCHAR(100),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on vector column for similarity search
CREATE INDEX idx_sequences_vector ON dna_similarity.sequences 
USING VECTOR(sequence_vector) WITH (type = 'HNSW');

-- Create index on class for filtering
CREATE INDEX idx_sequences_class ON dna_similarity.sequences (dna_class);

-- Create models table for model management
CREATE TABLE dna_similarity.models (
    id INTEGER IDENTITY PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    config_json VARCHAR(MAX),  -- JSON configuration
    model_path VARCHAR(500),   -- Path to serialized model
    vectorization_strategy VARCHAR(100),
    algorithm_name VARCHAR(100),
    algorithm_params VARCHAR(MAX),  -- JSON parameters
    training_data_size INTEGER,
    accuracy DOUBLE,
    precision_score DOUBLE,
    recall_score DOUBLE,
    f1_score DOUBLE,
    status VARCHAR(50) DEFAULT 'ACTIVE',
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trained_date TIMESTAMP,
    deployed_date TIMESTAMP
);

-- Create predictions table for audit trail
CREATE TABLE dna_similarity.predictions (
    id INTEGER IDENTITY PRIMARY KEY,
    model_id INTEGER REFERENCES dna_similarity.models(id),
    input_sequence VARCHAR(MAX) NOT NULL,
    predicted_class INTEGER,
    predicted_class_name VARCHAR(100),
    confidence_score DOUBLE,
    probabilities VARCHAR(MAX),  -- JSON with all class probabilities
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms INTEGER
);

-- Create similarity searches table for analytics
CREATE TABLE dna_similarity.similarity_searches (
    id INTEGER IDENTITY PRIMARY KEY,
    query_sequence VARCHAR(MAX) NOT NULL,
    query_vector VECTOR(DOUBLE, 768),
    top_k INTEGER DEFAULT 5,
    search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms INTEGER
);

-- Create similarity results table
CREATE TABLE dna_similarity.similarity_results (
    id INTEGER IDENTITY PRIMARY KEY,
    search_id INTEGER REFERENCES dna_similarity.similarity_searches(id),
    sequence_id INTEGER REFERENCES dna_similarity.sequences(id),
    similarity_score DOUBLE,
    rank_position INTEGER
);

-- Insert sample DNA class mappings
INSERT INTO dna_similarity.class_mappings (class_id, class_name, description) VALUES
(0, 'G protein coupled receptors', 'Cell membrane proteins that detect signaling molecules'),
(1, 'tyrosine kinase', 'Enzymes that transfer phosphate groups to tyrosine residues'),
(2, 'tyrosine phosphatase', 'Enzymes that remove phosphate groups from tyrosine residues'),
(3, 'synthetase', 'Enzymes that catalyze synthesis with ATP energy'),
(4, 'synthase', 'Enzymes that catalyze synthesis without ATP energy'),
(5, 'ion channel', 'Membrane proteins for selective ion transport'),
(6, 'transcription factor', 'Proteins that control gene transcription');

-- Create view for easy model performance monitoring
CREATE VIEW dna_similarity.model_performance AS
SELECT 
    m.model_name,
    m.model_version,
    m.algorithm_name,
    m.vectorization_strategy,
    m.accuracy,
    m.precision_score,
    m.recall_score,
    m.f1_score,
    m.training_data_size,
    COUNT(p.id) as prediction_count,
    AVG(p.confidence_score) as avg_confidence,
    m.status,
    m.created_date,
    m.deployed_date
FROM dna_similarity.models m
LEFT JOIN dna_similarity.predictions p ON m.id = p.model_id
GROUP BY m.id, m.model_name, m.model_version, m.algorithm_name, 
         m.vectorization_strategy, m.accuracy, m.precision_score,
         m.recall_score, m.f1_score, m.training_data_size,
         m.status, m.created_date, m.deployed_date;

-- Create stored procedure for similarity search
CREATE PROCEDURE dna_similarity.FindSimilarSequences(
    @query_sequence VARCHAR(MAX),
    @query_vector VARCHAR(MAX),  -- JSON array of vector values
    @top_k INTEGER = 5
)
AS
BEGIN
    -- Insert search record
    DECLARE @search_id INTEGER;
    INSERT INTO dna_similarity.similarity_searches 
    (query_sequence, query_vector, top_k, search_time)
    VALUES (@query_sequence, TO_VECTOR(@query_vector), @top_k, CURRENT_TIMESTAMP);
    
    SET @search_id = @@IDENTITY;
    
    -- Find similar sequences using vector dot product
    WITH SimilarSequences AS (
        SELECT 
            s.id,
            s.sequence,
            s.dna_class,
            s.dna_class_name,
            VECTOR_DOT_PRODUCT(s.sequence_vector, TO_VECTOR(@query_vector)) as similarity_score,
            ROW_NUMBER() OVER (ORDER BY VECTOR_DOT_PRODUCT(s.sequence_vector, TO_VECTOR(@query_vector)) DESC) as rank_pos
        FROM dna_similarity.sequences s
        WHERE s.sequence_vector IS NOT NULL
    )
    INSERT INTO dna_similarity.similarity_results (search_id, sequence_id, similarity_score, rank_position)
    SELECT @search_id, id, similarity_score, rank_pos
    FROM SimilarSequences
    WHERE rank_pos <= @top_k;
    
    -- Return results
    SELECT 
        s.sequence,
        s.dna_class,
        s.dna_class_name,
        sr.similarity_score,
        sr.rank_position
    FROM dna_similarity.similarity_results sr
    JOIN dna_similarity.sequences s ON sr.sequence_id = s.id
    WHERE sr.search_id = @search_id
    ORDER BY sr.rank_position;
END;

-- Create function for DNA sequence classification
CREATE FUNCTION dna_similarity.ClassifySequence(
    @sequence VARCHAR(MAX),
    @model_name VARCHAR(255) = 'dna_sequence_classifier'
)
RETURNS TABLE (
    predicted_class INTEGER,
    predicted_class_name VARCHAR(100),
    confidence_score DOUBLE,
    probabilities VARCHAR(MAX)
)
AS
EXTERNAL NAME 'IntegratedML.DNAClassifier.Predict';

-- Grant permissions for application user
GRANT SELECT, INSERT, UPDATE ON dna_similarity.* TO %IntegratedMLUser;
GRANT EXECUTE ON dna_similarity.FindSimilarSequences TO %IntegratedMLUser;
GRANT SELECT ON dna_similarity.ClassifySequence TO %IntegratedMLUser;

-- Create sample data (matching the demo)
INSERT INTO dna_similarity.sequences (sequence, k_mers, dna_class, dna_class_name) VALUES
('ATGTCTAAAAAGGAGAAAGACAAACTCACTATTCAATCAATCGCAATCAGCTGCATCATGGGCATGGTAGCCATTGCCATCGTCTGCTTCTTCCTGATGGTGATCCTCATCCTGCTGTGCTTCCTGGACAATGTGGGAGATGCTATGGAGGAGACCTCTAAGAACATCAAGCCCATCTATCAGGTGGACAATTTGTAG', 
 NULL, 0, 'G protein coupled receptors'),
('ATGAACTGTCCAGCCCCTGTGGAGATCTCCTATGAGAACATGCGTTTTCTGATATCTCACAACCCTACCAATGCTACTCTCAACAAGTTCACAGAGGAACTTAAGAAGTATGGAGTGACGACTTTGGTTCGAGTTTGTGATGCTACATATGATAAAGCTCCAGTTGAAAAAGAAGGAATCCACGTTCTAG',
 NULL, 2, 'tyrosine phosphatase'),
('ATGGCGCTGAACCCTTCCGGCACCATGGCCAAGTCCTACTTCATCGAGAACGCCATCAAGACCAAGGGCAAGCTGAAGAAGGACCTGAAGAACCCCATCAAGAAGGCCATCAAGGAGATCCTGAAGGACATCGAGAAGATCTACAAGGAGATCAAGAAGGACCTGAAGATCAAGAAGGACATCAAGAAGATCTAG',
 NULL, 6, 'transcription factor');

COMMIT;