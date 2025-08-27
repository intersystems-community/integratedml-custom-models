-- =============================================================================
-- Sales Forecasting Demo - IntegratedML Table Creation
-- =============================================================================
-- This script creates the necessary tables and views for the sales forecasting
-- demonstration in IntegratedML environment.
-- =============================================================================

-- Drop existing tables if they exist (for clean re-creation)
DROP TABLE IF EXISTS SalesForecast.SalesData;
DROP TABLE IF EXISTS SalesForecast.ExternalFactors;
DROP TABLE IF EXISTS SalesForecast.ModelPredictions;
DROP TABLE IF EXISTS SalesForecast.ModelMetrics;
DROP TABLE IF EXISTS SalesForecast.TrainingHistory;
DROP VIEW IF EXISTS SalesForecast.ForecastingView;
DROP SCHEMA IF EXISTS SalesForecast;

-- Create schema for sales forecasting
CREATE SCHEMA SalesForecast;

-- =============================================================================
-- Core Sales Data Table
-- =============================================================================
CREATE TABLE SalesForecast.SalesData (
    ID INTEGER IDENTITY PRIMARY KEY,
    SalesDate DATE NOT NULL,
    StoreID VARCHAR(20) NOT NULL,
    ProductCategory VARCHAR(50) NOT NULL,
    SalesAmount DECIMAL(12,2) NOT NULL,
    UnitsS INTEGER NOT NULL,
    
    -- Time-based features (for convenience)
    DayOfWeek INTEGER,
    Month INTEGER,
    Quarter INTEGER,
    Year INTEGER,
    WeekOfYear INTEGER,
    IsWeekend BOOLEAN,
    
    -- Calculated fields
    AveragePricePerUnit DECIMAL(10,2),
    
    -- Metadata
    DataSource VARCHAR(50) DEFAULT 'SYSTEM',
    CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_salesdata_date ON SalesForecast.SalesData(SalesDate);
CREATE INDEX idx_salesdata_store ON SalesForecast.SalesData(StoreID);
CREATE INDEX idx_salesdata_category ON SalesForecast.SalesData(ProductCategory);
CREATE INDEX idx_salesdata_composite ON SalesForecast.SalesData(SalesDate, StoreID, ProductCategory);

-- =============================================================================
-- External Factors Table
-- =============================================================================
CREATE TABLE SalesForecast.ExternalFactors (
    ID INTEGER IDENTITY PRIMARY KEY,
    FactorDate DATE NOT NULL,
    
    -- Weather data
    Temperature DECIMAL(5,2),
    Humidity DECIMAL(5,2),
    Precipitation DECIMAL(5,2),
    WeatherCondition VARCHAR(50),
    
    -- Economic indicators
    InflationRate DECIMAL(5,4),
    UnemploymentRate DECIMAL(5,2),
    ConsumerConfidenceIndex DECIMAL(7,2),
    GDP_Growth DECIMAL(5,2),
    
    -- Marketing factors
    PromotionActive BOOLEAN DEFAULT FALSE,
    PromotionType VARCHAR(50),
    PromotionDiscount DECIMAL(5,2),
    AdSpend DECIMAL(10,2),
    
    -- Calendar factors
    IsHoliday BOOLEAN DEFAULT FALSE,
    HolidayType VARCHAR(50),
    SchoolHoliday BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    DataSource VARCHAR(50) DEFAULT 'EXTERNAL',
    CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UpdatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_factors_date ON SalesForecast.ExternalFactors(FactorDate);
CREATE INDEX idx_factors_promotion ON SalesForecast.ExternalFactors(PromotionActive);
CREATE INDEX idx_factors_holiday ON SalesForecast.ExternalFactors(IsHoliday);

-- =============================================================================
-- Model Predictions Table
-- =============================================================================
CREATE TABLE SalesForecast.ModelPredictions (
    ID INTEGER IDENTITY PRIMARY KEY,
    ModelName VARCHAR(100) NOT NULL,
    ModelVersion VARCHAR(20) NOT NULL,
    PredictionDate DATE NOT NULL,
    ForecastDate DATE NOT NULL,
    StoreID VARCHAR(20) NOT NULL,
    ProductCategory VARCHAR(50) NOT NULL,
    
    -- Predictions
    PredictedSales DECIMAL(12,2) NOT NULL,
    ConfidenceLower DECIMAL(12,2),
    ConfidenceUpper DECIMAL(12,2),
    ConfidenceLevel DECIMAL(4,2) DEFAULT 0.95,
    
    -- Individual component predictions
    TrendComponent DECIMAL(12,2),
    SeasonalComponent DECIMAL(12,2),
    ExternalComponent DECIMAL(12,2),
    
    -- Model metadata
    PredictionScore DECIMAL(5,4),
    ModelUncertainty DECIMAL(5,4),
    FeatureImportance TEXT, -- JSON string
    
    -- Metadata
    CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    BatchID VARCHAR(50)
);

-- Create indexes
CREATE INDEX idx_predictions_model ON SalesForecast.ModelPredictions(ModelName, ModelVersion);
CREATE INDEX idx_predictions_date ON SalesForecast.ModelPredictions(PredictionDate, ForecastDate);
CREATE INDEX idx_predictions_store ON SalesForecast.ModelPredictions(StoreID, ProductCategory);

-- =============================================================================
-- Model Performance Metrics Table
-- =============================================================================
CREATE TABLE SalesForecast.ModelMetrics (
    ID INTEGER IDENTITY PRIMARY KEY,
    ModelName VARCHAR(100) NOT NULL,
    ModelVersion VARCHAR(20) NOT NULL,
    EvaluationDate DATE NOT NULL,
    EvaluationPeriod VARCHAR(50), -- 'daily', 'weekly', 'monthly'
    
    -- Accuracy metrics
    MAPE DECIMAL(7,4),
    MAE DECIMAL(12,2),
    RMSE DECIMAL(12,2),
    R_Squared DECIMAL(5,4),
    DirectionalAccuracy DECIMAL(5,4),
    
    -- Business metrics
    RevenueImpact DECIMAL(15,2),
    InventoryOptimization DECIMAL(15,2),
    ForecastBias DECIMAL(12,2),
    TrackingSignal DECIMAL(5,2),
    
    -- Statistical metrics
    TheilU DECIMAL(5,4),
    ForecastValueAdded DECIMAL(5,4),
    
    -- Coverage metrics (for interval predictions)
    CoverageRate_68 DECIMAL(5,4),
    CoverageRate_95 DECIMAL(5,4),
    CoverageRate_99 DECIMAL(5,4),
    
    -- Sample size
    SampleSize INTEGER,
    EvaluationPeriods INTEGER,
    
    -- Metadata
    MetricsData TEXT, -- JSON string with detailed metrics
    CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_metrics_model ON SalesForecast.ModelMetrics(ModelName, ModelVersion);
CREATE INDEX idx_metrics_date ON SalesForecast.ModelMetrics(EvaluationDate);
CREATE INDEX idx_metrics_mape ON SalesForecast.ModelMetrics(MAPE);

-- =============================================================================
-- Training History Table
-- =============================================================================
CREATE TABLE SalesForecast.TrainingHistory (
    ID INTEGER IDENTITY PRIMARY KEY,
    ModelName VARCHAR(100) NOT NULL,
    ModelVersion VARCHAR(20) NOT NULL,
    TrainingDate TIMESTAMP NOT NULL,
    
    -- Training data info
    TrainingStartDate DATE,
    TrainingEndDate DATE,
    TrainingRecords INTEGER,
    ValidationRecords INTEGER,
    
    -- Training configuration
    HyperParameters TEXT, -- JSON string
    Features TEXT, -- JSON array of feature names
    ModelConfig TEXT, -- JSON configuration
    
    -- Training results
    TrainingScore DECIMAL(5,4),
    ValidationScore DECIMAL(5,4),
    TrainingTime INTEGER, -- seconds
    
    -- Model artifacts
    ModelPath VARCHAR(500),
    ModelSize INTEGER, -- bytes
    
    -- Status
    TrainingStatus VARCHAR(20) DEFAULT 'COMPLETED', -- STARTED, COMPLETED, FAILED
    ErrorMessage TEXT,
    
    -- Metadata
    CreatedBy VARCHAR(100) DEFAULT USER,
    CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_training_model ON SalesForecast.TrainingHistory(ModelName, ModelVersion);
CREATE INDEX idx_training_date ON SalesForecast.TrainingHistory(TrainingDate);
CREATE INDEX idx_training_status ON SalesForecast.TrainingHistory(TrainingStatus);

-- =============================================================================
-- Comprehensive Forecasting View
-- =============================================================================
CREATE VIEW SalesForecast.ForecastingView AS
SELECT 
    s.SalesDate,
    s.StoreID,
    s.ProductCategory,
    s.SalesAmount AS ActualSales,
    s.UnitsS AS ActualUnits,
    s.DayOfWeek,
    s.Month,
    s.Quarter,
    s.Year,
    s.IsWeekend,
    
    -- External factors
    e.Temperature,
    e.Humidity,
    e.WeatherCondition,
    e.InflationRate,
    e.UnemploymentRate,
    e.PromotionActive,
    e.PromotionDiscount,
    e.IsHoliday,
    e.HolidayType,
    
    -- Lag features (previous day sales)
    LAG(s.SalesAmount, 1) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    ) AS Sales_Lag1,
    
    LAG(s.SalesAmount, 7) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    ) AS Sales_Lag7,
    
    LAG(s.SalesAmount, 30) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    ) AS Sales_Lag30,
    
    -- Rolling averages
    AVG(s.SalesAmount) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS Sales_MA7,
    
    AVG(s.SalesAmount) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS Sales_MA30,
    
    -- Growth rates
    (s.SalesAmount - LAG(s.SalesAmount, 1) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    )) / NULLIF(LAG(s.SalesAmount, 1) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    ), 0) * 100 AS GrowthRate_1D,
    
    (s.SalesAmount - LAG(s.SalesAmount, 7) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    )) / NULLIF(LAG(s.SalesAmount, 7) OVER (
        PARTITION BY s.StoreID, s.ProductCategory 
        ORDER BY s.SalesDate
    ), 0) * 100 AS GrowthRate_7D

FROM SalesForecast.SalesData s
LEFT JOIN SalesForecast.ExternalFactors e ON s.SalesDate = e.FactorDate
WHERE s.SalesDate >= DATEADD(year, -2, CURRENT_DATE); -- Last 2 years

-- =============================================================================
-- Data Quality and Constraints
-- =============================================================================

-- Add constraints
ALTER TABLE SalesForecast.SalesData 
ADD CONSTRAINT chk_sales_amount CHECK (SalesAmount >= 0);

ALTER TABLE SalesForecast.SalesData 
ADD CONSTRAINT chk_units_sold CHECK (UnitsS >= 0);

ALTER TABLE SalesForecast.ExternalFactors 
ADD CONSTRAINT chk_temperature CHECK (Temperature BETWEEN -50 AND 60);

ALTER TABLE SalesForecast.ExternalFactors 
ADD CONSTRAINT chk_humidity CHECK (Humidity BETWEEN 0 AND 100);

ALTER TABLE SalesForecast.ExternalFactors 
ADD CONSTRAINT chk_promotion_discount CHECK (PromotionDiscount BETWEEN 0 AND 1);

ALTER TABLE SalesForecast.ModelPredictions 
ADD CONSTRAINT chk_confidence_level CHECK (ConfidenceLevel BETWEEN 0 AND 1);

ALTER TABLE SalesForecast.ModelMetrics 
ADD CONSTRAINT chk_r_squared CHECK (R_Squared BETWEEN 0 AND 1);

-- =============================================================================
-- Sample Data Population (for demonstration)
-- =============================================================================

-- Insert sample stores and categories
INSERT INTO SalesForecast.SalesData (
    SalesDate, StoreID, ProductCategory, SalesAmount, UnitsS,
    DayOfWeek, Month, Quarter, Year, WeekOfYear, IsWeekend, AveragePricePerUnit
)
VALUES 
    ('2023-01-01', 'STORE001', 'Electronics', 15000.00, 50, 1, 1, 1, 2023, 1, FALSE, 300.00),
    ('2023-01-01', 'STORE001', 'Clothing', 8000.00, 160, 1, 1, 1, 2023, 1, FALSE, 50.00),
    ('2023-01-01', 'STORE002', 'Electronics', 12000.00, 40, 1, 1, 1, 2023, 1, FALSE, 300.00),
    ('2023-01-02', 'STORE001', 'Electronics', 16000.00, 55, 2, 1, 1, 2023, 1, FALSE, 290.91),
    ('2023-01-02', 'STORE001', 'Clothing', 9000.00, 180, 2, 1, 1, 2023, 1, FALSE, 50.00);

-- Insert sample external factors
INSERT INTO SalesForecast.ExternalFactors (
    FactorDate, Temperature, Humidity, WeatherCondition, 
    InflationRate, UnemploymentRate, ConsumerConfidenceIndex,
    PromotionActive, IsHoliday
)
VALUES 
    ('2023-01-01', 5.5, 65.0, 'Clear', 0.035, 5.2, 102.5, FALSE, TRUE), -- New Year's Day
    ('2023-01-02', 7.2, 70.0, 'Cloudy', 0.035, 5.2, 102.5, FALSE, FALSE);

-- =============================================================================
-- Comments and Documentation
-- =============================================================================

-- Add table comments
COMMENT ON TABLE SalesForecast.SalesData IS 'Core sales transaction data with time-based features for forecasting';
COMMENT ON TABLE SalesForecast.ExternalFactors IS 'External factors affecting sales including weather, economic indicators, and marketing activities';
COMMENT ON TABLE SalesForecast.ModelPredictions IS 'Model predictions with confidence intervals and component breakdowns';
COMMENT ON TABLE SalesForecast.ModelMetrics IS 'Model performance metrics and business impact measures';
COMMENT ON TABLE SalesForecast.TrainingHistory IS 'History of model training sessions with configurations and results';

-- Add column comments for key fields
COMMENT ON COLUMN SalesForecast.SalesData.SalesAmount IS 'Total sales amount in local currency';
COMMENT ON COLUMN SalesForecast.SalesData.UnitsS IS 'Number of units sold';
COMMENT ON COLUMN SalesForecast.ModelPredictions.PredictedSales IS 'Model predicted sales amount';
COMMENT ON COLUMN SalesForecast.ModelPredictions.ConfidenceLower IS 'Lower bound of confidence interval';
COMMENT ON COLUMN SalesForecast.ModelPredictions.ConfidenceUpper IS 'Upper bound of confidence interval';

-- =============================================================================
-- Stored Procedures for Common Operations
-- =============================================================================

-- Procedure to get latest model performance
CREATE OR REPLACE PROCEDURE SalesForecast.GetLatestModelPerformance(
    IN model_name VARCHAR(100) DEFAULT NULL,
    OUT result_cursor CURSOR
)
BEGIN
    IF model_name IS NULL THEN
        OPEN result_cursor FOR
        SELECT 
            ModelName,
            ModelVersion,
            EvaluationDate,
            MAPE,
            MAE,
            RMSE,
            R_Squared,
            DirectionalAccuracy,
            RevenueImpact
        FROM SalesForecast.ModelMetrics
        WHERE EvaluationDate = (
            SELECT MAX(EvaluationDate) 
            FROM SalesForecast.ModelMetrics m2 
            WHERE m2.ModelName = SalesForecast.ModelMetrics.ModelName
        )
        ORDER BY MAPE ASC;
    ELSE
        OPEN result_cursor FOR
        SELECT 
            ModelName,
            ModelVersion,
            EvaluationDate,
            MAPE,
            MAE,
            RMSE,
            R_Squared,
            DirectionalAccuracy,
            RevenueImpact
        FROM SalesForecast.ModelMetrics
        WHERE ModelName = model_name
        AND EvaluationDate = (
            SELECT MAX(EvaluationDate) 
            FROM SalesForecast.ModelMetrics 
            WHERE ModelName = model_name
        );
    END IF;
END;

-- Procedure to calculate data quality metrics
CREATE OR REPLACE PROCEDURE SalesForecast.CalculateDataQuality(
    IN start_date DATE DEFAULT NULL,
    IN end_date DATE DEFAULT NULL
)
BEGIN
    DECLARE total_records INTEGER;
    DECLARE missing_sales INTEGER;
    DECLARE missing_external INTEGER;
    DECLARE quality_score DECIMAL(5,2);
    
    -- Set default date range if not provided
    IF start_date IS NULL THEN
        SET start_date = DATEADD(month, -3, CURRENT_DATE);
    END IF;
    
    IF end_date IS NULL THEN
        SET end_date = CURRENT_DATE;
    END IF;
    
    -- Calculate data quality metrics
    SELECT COUNT(*) INTO total_records
    FROM SalesForecast.SalesData
    WHERE SalesDate BETWEEN start_date AND end_date;
    
    SELECT COUNT(*) INTO missing_sales
    FROM SalesForecast.SalesData
    WHERE SalesDate BETWEEN start_date AND end_date
    AND (SalesAmount IS NULL OR UnitsS IS NULL);
    
    SELECT COUNT(*) INTO missing_external
    FROM SalesForecast.SalesData s
    LEFT JOIN SalesForecast.ExternalFactors e ON s.SalesDate = e.FactorDate
    WHERE s.SalesDate BETWEEN start_date AND end_date
    AND e.FactorDate IS NULL;
    
    -- Calculate overall quality score
    SET quality_score = (1.0 - (missing_sales + missing_external) / total_records) * 100;
    
    -- Return results
    SELECT 
        start_date AS evaluation_start,
        end_date AS evaluation_end,
        total_records,
        missing_sales,
        missing_external,
        quality_score,
        CASE 
            WHEN quality_score >= 95 THEN 'Excellent'
            WHEN quality_score >= 90 THEN 'Good'
            WHEN quality_score >= 80 THEN 'Fair'
            ELSE 'Poor'
        END AS quality_rating,
        CURRENT_TIMESTAMP AS evaluation_time;
END;

-- =============================================================================
-- Grant Permissions (for multi-user environment)
-- =============================================================================

-- Grant permissions to forecast analysts
-- GRANT SELECT, INSERT, UPDATE ON SalesForecast.* TO ForecastAnalysts;
-- GRANT EXECUTE ON SalesForecast.GetLatestModelPerformance TO ForecastAnalysts;
-- GRANT EXECUTE ON SalesForecast.CalculateDataQuality TO ForecastAnalysts;

-- Grant read-only access to business users
-- GRANT SELECT ON SalesForecast.ForecastingView TO BusinessUsers;
-- GRANT SELECT ON SalesForecast.ModelMetrics TO BusinessUsers;
-- GRANT SELECT ON SalesForecast.ModelPredictions TO BusinessUsers;

PRINT 'Sales Forecasting database schema created successfully!';
PRINT 'Tables created: SalesData, ExternalFactors, ModelPredictions, ModelMetrics, TrainingHistory';
PRINT 'Views created: ForecastingView';
PRINT 'Stored procedures created: GetLatestModelPerformance, CalculateDataQuality';