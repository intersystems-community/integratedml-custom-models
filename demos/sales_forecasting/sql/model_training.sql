-- =============================================================================
-- Sales Forecasting Demo - IntegratedML Model Training
-- =============================================================================
-- This script demonstrates how to train and deploy the hybrid forecasting model
-- using IntegratedML SQL commands and integration with Python models.
-- =============================================================================

-- =============================================================================
-- Model Configuration and Registration
-- =============================================================================

-- Register the hybrid forecasting model with IntegratedML
CREATE OR REPLACE MODEL SalesForecast.HybridForecasting
PREDICTING (SalesAmount)
FROM SalesForecast.ForecastingView
USING {
    "pathtoregressors": "/opt/irisapp/demos/sales_forecasting/iris_models",
    
    "iscmodelsdisabled": 1,
    "user_params": {
        "prophet_config": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": true,
            "weekly_seasonality": true,
            "daily_seasonality": false,
            "holidays": true,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "n_changepoints": 25
        },
        "lightgbm_config": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42
        },
        "ensemble_config": {
            "prophet_weight": 0.6,
            "lightgbm_weight": 0.4,
            "use_dynamic_weighting": true,
            "horizon_weights": {
                "short_term": {"days": 7, "prophet_weight": 0.4, "lightgbm_weight": 0.6},
                "medium_term": {"days": 30, "prophet_weight": 0.6, "lightgbm_weight": 0.4},
                "long_term": {"days": 90, "prophet_weight": 0.8, "lightgbm_weight": 0.2}
            }
        },
        "feature_engineering": {
            "lag_features": [1, 7, 14, 30],
            "rolling_features": [7, 14, 30],
            "seasonal_features": true,
            "holiday_features": true,
            "weather_features": true,
            "economic_features": true
        }
    }
};

-- =============================================================================
-- Training Data Preparation
-- =============================================================================

-- Create training view with proper feature engineering
CREATE OR REPLACE VIEW SalesForecast.TrainingData AS
SELECT 
    SalesDate,
    StoreID,
    ProductCategory,
    SalesAmount,
    ActualUnits,
    
    -- Time features
    DayOfWeek,
    Month,
    Quarter,
    Year,
    IsWeekend,
    
    -- External factors
    Temperature,
    Humidity,
    WeatherCondition,
    InflationRate,
    UnemploymentRate,
    PromotionActive,
    PromotionDiscount,
    IsHoliday,
    HolidayType,
    
    -- Lag features
    Sales_Lag1,
    Sales_Lag7,
    Sales_Lag30,
    
    -- Rolling features
    Sales_MA7,
    Sales_MA30,
    
    -- Growth features
    GrowthRate_1D,
    GrowthRate_7D,
    
    -- Additional engineered features
    CASE 
        WHEN Month IN (11, 12) THEN 1 
        ELSE 0 
    END AS IsHolidaySeason,
    
    CASE 
        WHEN Month IN (6, 7, 8) THEN 1 
        ELSE 0 
    END AS IsSummer,
    
    CASE 
        WHEN DayOfWeek IN (1, 7) THEN 1 
        ELSE 0 
    END AS IsWeekendBinary,
    
    -- Interaction features
    Temperature * PromotionActive AS TempPromoInteraction,
    IsHoliday * PromotionActive AS HolidayPromoInteraction,
    
    -- Row number for splitting
    ROW_NUMBER() OVER (
        PARTITION BY StoreID, ProductCategory 
        ORDER BY SalesDate
    ) AS RowNum

FROM SalesForecast.ForecastingView
WHERE SalesDate >= DATEADD(year, -2, CURRENT_DATE)
AND SalesAmount IS NOT NULL
AND Sales_Lag1 IS NOT NULL; -- Ensure we have lag features

-- =============================================================================
-- Model Training Execution
-- =============================================================================

-- Start training session
BEGIN;

-- Log training start
INSERT INTO SalesForecast.TrainingHistory (
    ModelName,
    ModelVersion,
    TrainingDate,
    TrainingStartDate,
    TrainingEndDate,
    TrainingStatus,
    HyperParameters,
    Features
) VALUES (
    'HybridForecasting',
    '1.0',
    CURRENT_TIMESTAMP,
    (SELECT MIN(SalesDate) FROM SalesForecast.TrainingData),
    (SELECT MAX(SalesDate) FROM SalesForecast.TrainingData),
    'STARTED',
    '{
        "prophet_seasonality_mode": "multiplicative",
        "lightgbm_num_leaves": 31,
        "lightgbm_learning_rate": 0.05,
        "ensemble_prophet_weight": 0.6,
        "feature_lag_periods": [1, 7, 30],
        "rolling_window_sizes": [7, 30]
    }',
    '[
        "DayOfWeek", "Month", "Quarter", "IsWeekend",
        "Temperature", "Humidity", "InflationRate", "UnemploymentRate",
        "PromotionActive", "PromotionDiscount", "IsHoliday",
        "Sales_Lag1", "Sales_Lag7", "Sales_Lag30",
        "Sales_MA7", "Sales_MA30", "GrowthRate_1D", "GrowthRate_7D",
        "IsHolidaySeason", "IsSummer", "TempPromoInteraction", "HolidayPromoInteraction"
    ]'
);

-- Train the model using IntegratedML
TRAIN MODEL SalesForecast.HybridForecasting 
FROM SalesForecast.TrainingData
WHERE RowNum <= (
    SELECT MAX(RowNum) * 0.8 
    FROM SalesForecast.TrainingData
); -- Use 80% for training

-- Update training completion status
UPDATE SalesForecast.TrainingHistory 
SET 
    TrainingStatus = 'COMPLETED',
    TrainingRecords = (
        SELECT COUNT(*) 
        FROM SalesForecast.TrainingData 
        WHERE RowNum <= (SELECT MAX(RowNum) * 0.8 FROM SalesForecast.TrainingData)
    ),
    ValidationRecords = (
        SELECT COUNT(*) 
        FROM SalesForecast.TrainingData 
        WHERE RowNum > (SELECT MAX(RowNum) * 0.8 FROM SalesForecast.TrainingData)
    )
WHERE ModelName = 'HybridForecasting' 
AND ModelVersion = '1.0'
AND TrainingDate = (
    SELECT MAX(TrainingDate) 
    FROM SalesForecast.TrainingHistory 
    WHERE ModelName = 'HybridForecasting'
);

COMMIT;

-- =============================================================================
-- Model Validation and Testing
-- =============================================================================

-- Create validation predictions
CREATE OR REPLACE TEMPORARY VIEW ValidationPredictions AS
SELECT 
    v.*,
    p.SalesAmount AS PredictedSales,
    p.confidence_lower AS ConfidenceLower,
    p.confidence_upper AS ConfidenceUpper
FROM SalesForecast.TrainingData v
JOIN (
    SELECT 
        SalesDate,
        StoreID,
        ProductCategory,
        PREDICT(SalesForecast.HybridForecasting) AS SalesAmount,
        PREDICT(SalesForecast.HybridForecasting, 'confidence_lower') AS confidence_lower,
        PREDICT(SalesForecast.HybridForecasting, 'confidence_upper') AS confidence_upper
    FROM SalesForecast.TrainingData
    WHERE RowNum > (SELECT MAX(RowNum) * 0.8 FROM SalesForecast.TrainingData)
) p ON v.SalesDate = p.SalesDate 
    AND v.StoreID = p.StoreID 
    AND v.ProductCategory = p.ProductCategory
WHERE v.RowNum > (SELECT MAX(RowNum) * 0.8 FROM SalesForecast.TrainingData);

-- Calculate validation metrics
INSERT INTO SalesForecast.ModelMetrics (
    ModelName,
    ModelVersion,
    EvaluationDate,
    EvaluationPeriod,
    MAPE,
    MAE,
    RMSE,
    R_Squared,
    DirectionalAccuracy,
    ForecastBias,
    TrackingSignal,
    SampleSize,
    EvaluationPeriods
)
SELECT 
    'HybridForecasting' AS ModelName,
    '1.0' AS ModelVersion,
    CURRENT_DATE AS EvaluationDate,
    'validation' AS EvaluationPeriod,
    
    -- MAPE (Mean Absolute Percentage Error)
    AVG(ABS((SalesAmount - PredictedSales) / NULLIF(SalesAmount, 0))) * 100 AS MAPE,
    
    -- MAE (Mean Absolute Error)
    AVG(ABS(SalesAmount - PredictedSales)) AS MAE,
    
    -- RMSE (Root Mean Square Error)
    SQRT(AVG(POWER(SalesAmount - PredictedSales, 2))) AS RMSE,
    
    -- R-squared
    1 - (
        SUM(POWER(SalesAmount - PredictedSales, 2)) / 
        NULLIF(SUM(POWER(SalesAmount - AVG(SalesAmount) OVER(), 2)), 0)
    ) AS R_Squared,
    
    -- Directional Accuracy
    AVG(CASE 
        WHEN SIGN(SalesAmount - LAG(SalesAmount) OVER (ORDER BY SalesDate)) = 
             SIGN(PredictedSales - LAG(PredictedSales) OVER (ORDER BY SalesDate))
        THEN 1.0 
        ELSE 0.0 
    END) AS DirectionalAccuracy,
    
    -- Forecast Bias
    AVG(SalesAmount - PredictedSales) AS ForecastBias,
    
    -- Tracking Signal
    SUM(SalesAmount - PredictedSales) / NULLIF(SUM(ABS(SalesAmount - PredictedSales)), 0) AS TrackingSignal,
    
    COUNT(*) AS SampleSize,
    COUNT(DISTINCT SalesDate) AS EvaluationPeriods

FROM ValidationPredictions;

-- =============================================================================
-- Model Deployment and Scoring
-- =============================================================================

-- Create scoring function for real-time predictions
CREATE OR REPLACE FUNCTION SalesForecast.ScoreForecasting(
    in_sales_date DATE,
    in_store_id VARCHAR(20),
    in_product_category VARCHAR(50),
    in_temperature DECIMAL(5,2) DEFAULT NULL,
    in_humidity DECIMAL(5,2) DEFAULT NULL,
    in_promotion_active BOOLEAN DEFAULT FALSE,
    in_promotion_discount DECIMAL(5,2) DEFAULT 0,
    in_is_holiday BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    predicted_sales DECIMAL(12,2),
    confidence_lower DECIMAL(12,2),
    confidence_upper DECIMAL(12,2),
    prediction_date TIMESTAMP
)
AS $$
BEGIN
    RETURN QUERY
    WITH InputData AS (
        SELECT 
            in_sales_date AS SalesDate,
            in_store_id AS StoreID,
            in_product_category AS ProductCategory,
            EXTRACT(DOW FROM in_sales_date) AS DayOfWeek,
            EXTRACT(MONTH FROM in_sales_date) AS Month,
            EXTRACT(QUARTER FROM in_sales_date) AS Quarter,
            EXTRACT(YEAR FROM in_sales_date) AS Year,
            CASE WHEN EXTRACT(DOW FROM in_sales_date) IN (0, 6) THEN 1 ELSE 0 END AS IsWeekend,
            COALESCE(in_temperature, 20.0) AS Temperature,
            COALESCE(in_humidity, 50.0) AS Humidity,
            in_promotion_active AS PromotionActive,
            in_promotion_discount AS PromotionDiscount,
            in_is_holiday AS IsHoliday,
            
            -- Get recent sales for lag features
            (SELECT SalesAmount FROM SalesForecast.SalesData 
             WHERE StoreID = in_store_id 
             AND ProductCategory = in_product_category 
             AND SalesDate = in_sales_date - 1) AS Sales_Lag1,
             
            (SELECT SalesAmount FROM SalesForecast.SalesData 
             WHERE StoreID = in_store_id 
             AND ProductCategory = in_product_category 
             AND SalesDate = in_sales_date - 7) AS Sales_Lag7,
             
            (SELECT AVG(SalesAmount) FROM SalesForecast.SalesData 
             WHERE StoreID = in_store_id 
             AND ProductCategory = in_product_category 
             AND SalesDate BETWEEN in_sales_date - 7 AND in_sales_date - 1) AS Sales_MA7
    )
    SELECT 
        PREDICT(SalesForecast.HybridForecasting, i) AS predicted_sales,
        PREDICT(SalesForecast.HybridForecasting, i, 'confidence_lower') AS confidence_lower,
        PREDICT(SalesForecast.HybridForecasting, i, 'confidence_upper') AS confidence_upper,
        CURRENT_TIMESTAMP AS prediction_date
    FROM InputData i;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Batch Forecasting Procedure
-- =============================================================================

CREATE OR REPLACE PROCEDURE SalesForecast.GenerateBatchForecast(
    IN forecast_start_date DATE,
    IN forecast_end_date DATE,
    IN store_list VARCHAR(1000) DEFAULT NULL, -- Comma-separated store IDs
    IN category_list VARCHAR(1000) DEFAULT NULL -- Comma-separated categories
)
AS $$
DECLARE
    batch_id VARCHAR(50);
    forecast_cursor CURSOR FOR
        SELECT DISTINCT StoreID, ProductCategory
        FROM SalesForecast.SalesData
        WHERE (store_list IS NULL OR StoreID = ANY(STRING_TO_ARRAY(store_list, ',')))
        AND (category_list IS NULL OR ProductCategory = ANY(STRING_TO_ARRAY(category_list, ',')));
    
    store_rec RECORD;
    current_date DATE;
BEGIN
    -- Generate unique batch ID
    batch_id := 'BATCH_' || TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    
    -- Log batch start
    INSERT INTO SalesForecast.TrainingHistory (
        ModelName,
        ModelVersion,
        TrainingDate,
        TrainingStatus,
        ErrorMessage
    ) VALUES (
        'HybridForecasting',
        '1.0',
        CURRENT_TIMESTAMP,
        'BATCH_FORECASTING',
        'Batch ID: ' || batch_id
    );
    
    -- Generate forecasts for each store/category combination
    FOR store_rec IN forecast_cursor LOOP
        current_date := forecast_start_date;
        
        WHILE current_date <= forecast_end_date LOOP
            -- Generate forecast using the trained model
            INSERT INTO SalesForecast.ModelPredictions (
                ModelName,
                ModelVersion,
                PredictionDate,
                ForecastDate,
                StoreID,
                ProductCategory,
                PredictedSales,
                ConfidenceLower,
                ConfidenceUpper,
                BatchID
            )
            SELECT 
                'HybridForecasting',
                '1.0',
                CURRENT_DATE,
                current_date,
                store_rec.StoreID,
                store_rec.ProductCategory,
                predicted_sales,
                confidence_lower,
                confidence_upper,
                batch_id
            FROM SalesForecast.ScoreForecasting(
                current_date,
                store_rec.StoreID,
                store_rec.ProductCategory
            );
            
            current_date := current_date + 1;
        END LOOP;
    END LOOP;
    
    -- Update batch completion
    UPDATE SalesForecast.TrainingHistory 
    SET TrainingStatus = 'BATCH_COMPLETED'
    WHERE ErrorMessage = 'Batch ID: ' || batch_id;
    
    RAISE NOTICE 'Batch forecast completed. Batch ID: %', batch_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Model Performance Monitoring
-- =============================================================================

-- Create monitoring view for ongoing performance tracking
CREATE OR REPLACE VIEW SalesForecast.ModelPerformanceMonitor AS
WITH RecentPredictions AS (
    SELECT 
        p.ModelName,
        p.ModelVersion,
        p.PredictionDate,
        p.ForecastDate,
        p.StoreID,
        p.ProductCategory,
        p.PredictedSales,
        s.SalesAmount AS ActualSales,
        ABS(s.SalesAmount - p.PredictedSales) AS AbsError,
        ABS((s.SalesAmount - p.PredictedSales) / NULLIF(s.SalesAmount, 0)) AS AbsPercentError
    FROM SalesForecast.ModelPredictions p
    JOIN SalesForecast.SalesData s ON 
        p.ForecastDate = s.SalesDate
        AND p.StoreID = s.StoreID
        AND p.ProductCategory = s.ProductCategory
    WHERE p.PredictionDate >= CURRENT_DATE - 30 -- Last 30 days of predictions
)
SELECT 
    ModelName,
    ModelVersion,
    COUNT(*) AS TotalPredictions,
    AVG(AbsPercentError) * 100 AS MAPE,
    AVG(AbsError) AS MAE,
    SQRT(AVG(POWER(ActualSales - PredictedSales, 2))) AS RMSE,
    AVG(ActualSales - PredictedSales) AS Bias,
    STDDEV(ActualSales - PredictedSales) AS ErrorStdDev,
    MIN(PredictionDate) AS EarliestPrediction,
    MAX(PredictionDate) AS LatestPrediction,
    CURRENT_TIMESTAMP AS LastUpdated
FROM RecentPredictions
GROUP BY ModelName, ModelVersion;

-- =============================================================================
-- Automated Model Retraining Trigger
-- =============================================================================

-- Create procedure for automated model retraining
CREATE OR REPLACE PROCEDURE SalesForecast.CheckModelPerformance()
AS $$
DECLARE
    current_mape DECIMAL(7,4);
    baseline_mape DECIMAL(7,4);
    performance_degradation DECIMAL(5,2);
BEGIN
    -- Get current model performance
    SELECT MAPE INTO current_mape
    FROM SalesForecast.ModelPerformanceMonitor
    WHERE ModelName = 'HybridForecasting'
    AND ModelVersion = '1.0';
    
    -- Get baseline performance (from initial validation)
    SELECT MAPE INTO baseline_mape
    FROM SalesForecast.ModelMetrics
    WHERE ModelName = 'HybridForecasting'
    AND ModelVersion = '1.0'
    AND EvaluationPeriod = 'validation';
    
    -- Calculate performance degradation
    performance_degradation := (current_mape - baseline_mape) / baseline_mape * 100;
    
    -- If performance has degraded significantly, trigger retraining
    IF performance_degradation > 20 THEN -- 20% degradation threshold
        INSERT INTO SalesForecast.TrainingHistory (
            ModelName,
            ModelVersion,
            TrainingDate,
            TrainingStatus,
            ErrorMessage
        ) VALUES (
            'HybridForecasting',
            '1.1', -- New version
            CURRENT_TIMESTAMP,
            'RETRAINING_TRIGGERED',
            'Performance degradation: ' || performance_degradation || '%'
        );
        
        RAISE NOTICE 'Model retraining triggered due to performance degradation: %', performance_degradation;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Model Information and Metadata Queries
-- =============================================================================

-- View to show model information
CREATE OR REPLACE VIEW SalesForecast.ModelInfo AS
SELECT 
    'HybridForecasting' AS ModelName,
    '1.0' AS ModelVersion,
    'Hybrid Prophet + LightGBM forecasting model' AS Description,
    'demos.sales_forecasting.models.hybrid_forecasting_model.HybridForecastingModel' AS PythonClass,
    (SELECT COUNT(*) FROM SalesForecast.TrainingHistory WHERE ModelName = 'HybridForecasting') AS TrainingRuns,
    (SELECT MAX(TrainingDate) FROM SalesForecast.TrainingHistory WHERE ModelName = 'HybridForecasting') AS LastTrained,
    (SELECT COUNT(*) FROM SalesForecast.ModelPredictions WHERE ModelName = 'HybridForecasting') AS TotalPredictions,
    (SELECT COUNT(*) FROM SalesForecast.ModelMetrics WHERE ModelName = 'HybridForecasting') AS MetricEvaluations;

-- Example usage queries
/*

-- 1. Train the model
EXEC SalesForecast.HybridForecasting.TRAIN;

-- 2. Generate single forecast
SELECT * FROM SalesForecast.ScoreForecasting(
    '2024-01-15'::DATE,
    'STORE001',
    'Electronics',
    15.5,  -- temperature
    65.0,  -- humidity
    TRUE,  -- promotion active
    0.15,  -- 15% discount
    FALSE  -- not a holiday
);

-- 3. Generate batch forecasts for next 30 days
CALL SalesForecast.GenerateBatchForecast(
    CURRENT_DATE + 1,
    CURRENT_DATE + 30,
    'STORE001,STORE002',
    'Electronics,Clothing'
);

-- 4. Check model performance
SELECT * FROM SalesForecast.ModelPerformanceMonitor;

-- 5. View model information
SELECT * FROM SalesForecast.ModelInfo;

-- 6. Check for performance degradation
CALL SalesForecast.CheckModelPerformance();

*/

PRINT 'Sales Forecasting model training scripts created successfully!';
PRINT 'Model: HybridForecasting registered with IntegratedML';
PRINT 'Functions: ScoreForecasting for real-time predictions';
PRINT 'Procedures: GenerateBatchForecast, CheckModelPerformance';
PRINT 'Views: TrainingData, ModelPerformanceMonitor, ModelInfo';