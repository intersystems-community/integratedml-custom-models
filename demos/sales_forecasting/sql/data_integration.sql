-- =============================================================================
-- Sales Forecasting Demo - Data Integration and Deployment
-- =============================================================================
-- This script provides data integration utilities, ETL procedures, and 
-- deployment operations for the sales forecasting system.
-- =============================================================================

-- =============================================================================
-- Data Loading and ETL Procedures
-- =============================================================================

-- Procedure to load sales data from external CSV files
CREATE OR REPLACE PROCEDURE SalesForecast.LoadSalesDataFromCSV(
    IN csv_file_path VARCHAR(500),
    IN batch_size INTEGER DEFAULT 1000,
    IN validate_data BOOLEAN DEFAULT TRUE
)
AS $$
DECLARE
    loaded_records INTEGER := 0;
    validation_errors INTEGER := 0;
    batch_id VARCHAR(50);
BEGIN
    -- Generate batch ID for tracking
    batch_id := 'LOAD_' || TO_CHAR(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MISS');
    
    -- Create temporary staging table
    CREATE TEMPORARY TABLE temp_sales_staging (
        sales_date VARCHAR(20),
        store_id VARCHAR(20),
        product_category VARCHAR(50),
        sales_amount VARCHAR(20),
        units_sold VARCHAR(20),
        data_source VARCHAR(50) DEFAULT 'CSV_IMPORT'
    );
    
    -- Load data from CSV file (syntax may vary by database system)
    EXECUTE format('COPY temp_sales_staging (sales_date, store_id, product_category, sales_amount, units_sold) 
                   FROM %L WITH CSV HEADER', csv_file_path);
    
    GET DIAGNOSTICS loaded_records = ROW_COUNT;
    
    -- Data validation if requested
    IF validate_data THEN
        -- Check for invalid dates
        SELECT COUNT(*) INTO validation_errors
        FROM temp_sales_staging
        WHERE NOT sales_date::DATE BETWEEN '2020-01-01' AND CURRENT_DATE + 365;
        
        IF validation_errors > 0 THEN
            RAISE EXCEPTION 'Validation failed: % invalid dates found', validation_errors;
        END IF;
        
        -- Check for negative amounts
        SELECT COUNT(*) INTO validation_errors
        FROM temp_sales_staging
        WHERE sales_amount::DECIMAL < 0 OR units_sold::INTEGER < 0;
        
        IF validation_errors > 0 THEN
            RAISE EXCEPTION 'Validation failed: % negative values found', validation_errors;
        END IF;
    END IF;
    
    -- Insert validated data into main table
    INSERT INTO SalesForecast.SalesData (
        SalesDate,
        StoreID,
        ProductCategory,
        SalesAmount,
        UnitsS,
        DayOfWeek,
        Month,
        Quarter,
        Year,
        WeekOfYear,
        IsWeekend,
        AveragePricePerUnit,
        DataSource
    )
    SELECT 
        sales_date::DATE,
        store_id,
        product_category,
        sales_amount::DECIMAL(12,2),
        units_sold::INTEGER,
        EXTRACT(DOW FROM sales_date::DATE),
        EXTRACT(MONTH FROM sales_date::DATE),
        EXTRACT(QUARTER FROM sales_date::DATE),
        EXTRACT(YEAR FROM sales_date::DATE),
        EXTRACT(WEEK FROM sales_date::DATE),
        CASE WHEN EXTRACT(DOW FROM sales_date::DATE) IN (0, 6) THEN TRUE ELSE FALSE END,
        CASE WHEN units_sold::INTEGER > 0 THEN sales_amount::DECIMAL / units_sold::INTEGER ELSE 0 END,
        'CSV_' || batch_id
    FROM temp_sales_staging;
    
    -- Clean up
    DROP TABLE temp_sales_staging;
    
    RAISE NOTICE 'Successfully loaded % records from CSV file: %', loaded_records, csv_file_path;
END;
$$ LANGUAGE plpgsql;

-- Procedure to load external factors data
CREATE OR REPLACE PROCEDURE SalesForecast.LoadExternalFactors(
    IN data_source VARCHAR(50),
    IN start_date DATE DEFAULT NULL,
    IN end_date DATE DEFAULT NULL
)
AS $$
DECLARE
    weather_api_key VARCHAR(100);
    economic_api_key VARCHAR(100);
    current_date_iter DATE;
BEGIN
    -- Set default date range
    IF start_date IS NULL THEN
        start_date := CURRENT_DATE - 30;
    END IF;
    
    IF end_date IS NULL THEN
        end_date := CURRENT_DATE;
    END IF;
    
    current_date_iter := start_date;
    
    -- Load weather data (mock implementation - would integrate with real APIs)
    WHILE current_date_iter <= end_date LOOP
        INSERT INTO SalesForecast.ExternalFactors (
            FactorDate,
            Temperature,
            Humidity,
            WeatherCondition,
            DataSource
        )
        VALUES (
            current_date_iter,
            15 + (RANDOM() * 20), -- Mock temperature 15-35°C
            40 + (RANDOM() * 40), -- Mock humidity 40-80%
            CASE (RANDOM() * 4)::INTEGER
                WHEN 0 THEN 'Sunny'
                WHEN 1 THEN 'Cloudy'
                WHEN 2 THEN 'Rainy'
                ELSE 'Clear'
            END,
            data_source
        )
        ON CONFLICT (FactorDate) DO UPDATE SET
            Temperature = EXCLUDED.Temperature,
            Humidity = EXCLUDED.Humidity,
            WeatherCondition = EXCLUDED.WeatherCondition,
            UpdatedDate = CURRENT_TIMESTAMP;
        
        current_date_iter := current_date_iter + 1;
    END LOOP;
    
    RAISE NOTICE 'Loaded external factors for % days from % to %', 
                 (end_date - start_date + 1), start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Data Quality and Monitoring
-- =============================================================================

-- Data quality assessment procedure
CREATE OR REPLACE PROCEDURE SalesForecast.AssessDataQuality(
    OUT quality_report TEXT
)
AS $$
DECLARE
    total_sales_records INTEGER;
    missing_sales INTEGER;
    duplicate_sales INTEGER;
    outlier_sales INTEGER;
    data_gaps INTEGER;
    quality_score DECIMAL(5,2);
    issues TEXT := '';
BEGIN
    -- Count total sales records
    SELECT COUNT(*) INTO total_sales_records
    FROM SalesForecast.SalesData
    WHERE SalesDate >= CURRENT_DATE - 90;
    
    -- Check for missing critical fields
    SELECT COUNT(*) INTO missing_sales
    FROM SalesForecast.SalesData
    WHERE SalesDate >= CURRENT_DATE - 90
    AND (SalesAmount IS NULL OR UnitsS IS NULL OR StoreID IS NULL);
    
    -- Check for duplicates
    SELECT COUNT(*) - COUNT(DISTINCT(SalesDate, StoreID, ProductCategory)) INTO duplicate_sales
    FROM SalesForecast.SalesData
    WHERE SalesDate >= CURRENT_DATE - 90;
    
    -- Check for statistical outliers (sales > 3 standard deviations)
    WITH stats AS (
        SELECT 
            AVG(SalesAmount) AS mean_sales,
            STDDEV(SalesAmount) AS std_sales
        FROM SalesForecast.SalesData
        WHERE SalesDate >= CURRENT_DATE - 90
    )
    SELECT COUNT(*) INTO outlier_sales
    FROM SalesForecast.SalesData s, stats st
    WHERE s.SalesDate >= CURRENT_DATE - 90
    AND ABS(s.SalesAmount - st.mean_sales) > 3 * st.std_sales;
    
    -- Check for date gaps
    WITH date_series AS (
        SELECT generate_series(
            CURRENT_DATE - 90,
            CURRENT_DATE - 1,
            '1 day'::interval
        )::DATE AS expected_date
    ),
    missing_dates AS (
        SELECT d.expected_date
        FROM date_series d
        LEFT JOIN (
            SELECT DISTINCT SalesDate
            FROM SalesForecast.SalesData
            WHERE SalesDate >= CURRENT_DATE - 90
        ) s ON d.expected_date = s.SalesDate
        WHERE s.SalesDate IS NULL
    )
    SELECT COUNT(*) INTO data_gaps FROM missing_dates;
    
    -- Calculate quality score
    quality_score := GREATEST(0, 100 - (
        (missing_sales::DECIMAL / total_sales_records * 30) +
        (duplicate_sales::DECIMAL / total_sales_records * 20) +
        (outlier_sales::DECIMAL / total_sales_records * 15) +
        (data_gaps::DECIMAL / 90 * 35)
    ));
    
    -- Build issues list
    IF missing_sales > 0 THEN
        issues := issues || format('- %s records with missing critical fields\n', missing_sales);
    END IF;
    
    IF duplicate_sales > 0 THEN
        issues := issues || format('- %s duplicate records detected\n', duplicate_sales);
    END IF;
    
    IF outlier_sales > 0 THEN
        issues := issues || format('- %s statistical outliers identified\n', outlier_sales);
    END IF;
    
    IF data_gaps > 0 THEN
        issues := issues || format('- %s days with missing data\n', data_gaps);
    END IF;
    
    -- Generate quality report
    quality_report := format(
        'DATA QUALITY ASSESSMENT REPORT\n' ||
        '================================\n' ||
        'Assessment Date: %s\n' ||
        'Data Period: Last 90 days\n' ||
        'Total Records: %s\n\n' ||
        'QUALITY SCORE: %.1f/100\n\n' ||
        'ISSUES IDENTIFIED:\n%s\n' ||
        'RECOMMENDATIONS:\n' ||
        CASE 
            WHEN quality_score >= 90 THEN '- Data quality is excellent, no immediate action required\n'
            WHEN quality_score >= 80 THEN '- Data quality is good, monitor identified issues\n'
            WHEN quality_score >= 70 THEN '- Data quality is acceptable, address major issues\n'
            ELSE '- Data quality is poor, immediate remediation required\n'
        END ||
        CASE WHEN missing_sales > 0 THEN '- Investigate and fill missing data gaps\n' ELSE '' END ||
        CASE WHEN duplicate_sales > 0 THEN '- Remove duplicate records and fix data ingestion\n' ELSE '' END ||
        CASE WHEN outlier_sales > 0 THEN '- Review and validate outlier transactions\n' ELSE '' END,
        CURRENT_TIMESTAMP,
        total_sales_records,
        quality_score,
        COALESCE(NULLIF(issues, ''), '- No issues detected\n')
    );
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Model Deployment and API Integration
-- =============================================================================

-- Create REST API endpoint simulation for model scoring
CREATE OR REPLACE FUNCTION SalesForecast.APIScoreEndpoint(
    request_json TEXT
)
RETURNS TEXT
AS $$
DECLARE
    request_data JSONB;
    sales_date DATE;
    store_id VARCHAR(20);
    product_category VARCHAR(50);
    temperature DECIMAL(5,2);
    humidity DECIMAL(5,2);
    promotion_active BOOLEAN;
    promotion_discount DECIMAL(5,2);
    is_holiday BOOLEAN;
    prediction_result RECORD;
    response_json JSONB;
BEGIN
    -- Parse input JSON
    request_data := request_json::JSONB;
    
    -- Extract parameters
    sales_date := (request_data->>'sales_date')::DATE;
    store_id := request_data->>'store_id';
    product_category := request_data->>'product_category';
    temperature := COALESCE((request_data->>'temperature')::DECIMAL(5,2), 20.0);
    humidity := COALESCE((request_data->>'humidity')::DECIMAL(5,2), 50.0);
    promotion_active := COALESCE((request_data->>'promotion_active')::BOOLEAN, FALSE);
    promotion_discount := COALESCE((request_data->>'promotion_discount')::DECIMAL(5,2), 0.0);
    is_holiday := COALESCE((request_data->>'is_holiday')::BOOLEAN, FALSE);
    
    -- Get prediction
    SELECT * INTO prediction_result
    FROM SalesForecast.ScoreForecasting(
        sales_date,
        store_id,
        product_category,
        temperature,
        humidity,
        promotion_active,
        promotion_discount,
        is_holiday
    );
    
    -- Build response JSON
    response_json := jsonb_build_object(
        'status', 'success',
        'prediction', jsonb_build_object(
            'predicted_sales', prediction_result.predicted_sales,
            'confidence_lower', prediction_result.confidence_lower,
            'confidence_upper', prediction_result.confidence_upper,
            'prediction_date', prediction_result.prediction_date
        ),
        'request_parameters', jsonb_build_object(
            'sales_date', sales_date,
            'store_id', store_id,
            'product_category', product_category,
            'temperature', temperature,
            'humidity', humidity,
            'promotion_active', promotion_active,
            'promotion_discount', promotion_discount,
            'is_holiday', is_holiday
        )
    );
    
    RETURN response_json::TEXT;
    
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object(
            'status', 'error',
            'error_message', SQLERRM,
            'error_code', SQLSTATE
        )::TEXT;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Automated Maintenance and Cleanup
-- =============================================================================

-- Procedure for automated data archiving
CREATE OR REPLACE PROCEDURE SalesForecast.ArchiveOldData(
    IN retention_days INTEGER DEFAULT 730 -- 2 years default
)
AS $$
DECLARE
    cutoff_date DATE;
    archived_sales INTEGER;
    archived_predictions INTEGER;
    archived_metrics INTEGER;
BEGIN
    cutoff_date := CURRENT_DATE - retention_days;
    
    -- Archive old sales data
    CREATE TABLE IF NOT EXISTS SalesForecast.SalesDataArchive (LIKE SalesForecast.SalesData);
    
    INSERT INTO SalesForecast.SalesDataArchive
    SELECT * FROM SalesForecast.SalesData
    WHERE SalesDate < cutoff_date;
    
    GET DIAGNOSTICS archived_sales = ROW_COUNT;
    
    DELETE FROM SalesForecast.SalesData
    WHERE SalesDate < cutoff_date;
    
    -- Archive old predictions
    CREATE TABLE IF NOT EXISTS SalesForecast.ModelPredictionsArchive (LIKE SalesForecast.ModelPredictions);
    
    INSERT INTO SalesForecast.ModelPredictionsArchive
    SELECT * FROM SalesForecast.ModelPredictions
    WHERE PredictionDate < cutoff_date;
    
    GET DIAGNOSTICS archived_predictions = ROW_COUNT;
    
    DELETE FROM SalesForecast.ModelPredictions
    WHERE PredictionDate < cutoff_date;
    
    -- Archive old metrics
    CREATE TABLE IF NOT EXISTS SalesForecast.ModelMetricsArchive (LIKE SalesForecast.ModelMetrics);
    
    INSERT INTO SalesForecast.ModelMetricsArchive
    SELECT * FROM SalesForecast.ModelMetrics
    WHERE EvaluationDate < cutoff_date;
    
    GET DIAGNOSTICS archived_metrics = ROW_COUNT;
    
    DELETE FROM SalesForecast.ModelMetrics
    WHERE EvaluationDate < cutoff_date;
    
    RAISE NOTICE 'Archival complete. Archived: % sales records, % predictions, % metrics',
                 archived_sales, archived_predictions, archived_metrics;
END;
$$ LANGUAGE plpgsql;

-- Procedure for database maintenance
CREATE OR REPLACE PROCEDURE SalesForecast.MaintenanceCleanup()
AS $$
BEGIN
    -- Update table statistics
    ANALYZE SalesForecast.SalesData;
    ANALYZE SalesForecast.ExternalFactors;
    ANALYZE SalesForecast.ModelPredictions;
    ANALYZE SalesForecast.ModelMetrics;
    
    -- Reindex tables for performance
    REINDEX TABLE SalesForecast.SalesData;
    REINDEX TABLE SalesForecast.ExternalFactors;
    REINDEX TABLE SalesForecast.ModelPredictions;
    
    -- Clean up old training history (keep last 50 entries)
    DELETE FROM SalesForecast.TrainingHistory
    WHERE ID NOT IN (
        SELECT ID FROM SalesForecast.TrainingHistory
        ORDER BY TrainingDate DESC
        LIMIT 50
    );
    
    -- Vacuum tables to reclaim space
    VACUUM ANALYZE SalesForecast.SalesData;
    VACUUM ANALYZE SalesForecast.ExternalFactors;
    VACUUM ANALYZE SalesForecast.ModelPredictions;
    VACUUM ANALYZE SalesForecast.ModelMetrics;
    
    RAISE NOTICE 'Database maintenance completed successfully';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Business Intelligence Views
-- =============================================================================

-- Executive dashboard view
CREATE OR REPLACE VIEW SalesForecast.ExecutiveDashboard AS
WITH RecentPerformance AS (
    SELECT 
        AVG(CASE WHEN p.PredictedSales > 0 THEN 
            ABS((s.SalesAmount - p.PredictedSales) / p.PredictedSales) * 100 
            ELSE NULL END) AS CurrentMAPE,
        COUNT(*) AS RecentPredictions,
        SUM(s.SalesAmount) AS ActualRevenue,
        SUM(p.PredictedSales) AS ForecastRevenue
    FROM SalesForecast.ModelPredictions p
    JOIN SalesForecast.SalesData s ON 
        p.ForecastDate = s.SalesDate
        AND p.StoreID = s.StoreID
        AND p.ProductCategory = s.ProductCategory
    WHERE p.PredictionDate >= CURRENT_DATE - 7
),
TrendAnalysis AS (
    SELECT 
        StoreID,
        ProductCategory,
        AVG(SalesAmount) AS AvgSales,
        STDDEV(SalesAmount) AS SalesVolatility,
        (MAX(SalesAmount) - MIN(SalesAmount)) / AVG(SalesAmount) AS SalesRange
    FROM SalesForecast.SalesData
    WHERE SalesDate >= CURRENT_DATE - 30
    GROUP BY StoreID, ProductCategory
)
SELECT 
    CURRENT_DATE AS DashboardDate,
    rp.CurrentMAPE AS "Forecast Accuracy (%)",
    100 - rp.CurrentMAPE AS "Accuracy Score",
    rp.ActualRevenue AS "Actual Revenue (Last 7 Days)",
    rp.ForecastRevenue AS "Forecast Revenue (Last 7 Days)",
    (rp.ForecastRevenue - rp.ActualRevenue) / rp.ActualRevenue * 100 AS "Revenue Variance (%)",
    rp.RecentPredictions AS "Predictions Generated",
    (SELECT COUNT(DISTINCT StoreID) FROM SalesForecast.SalesData WHERE SalesDate >= CURRENT_DATE - 7) AS "Active Stores",
    (SELECT COUNT(DISTINCT ProductCategory) FROM SalesForecast.SalesData WHERE SalesDate >= CURRENT_DATE - 7) AS "Product Categories",
    (SELECT AVG(SalesVolatility) FROM TrendAnalysis) AS "Average Volatility",
    CASE 
        WHEN rp.CurrentMAPE <= 5 THEN 'Excellent'
        WHEN rp.CurrentMAPE <= 10 THEN 'Good'
        WHEN rp.CurrentMAPE <= 15 THEN 'Fair'
        ELSE 'Needs Improvement'
    END AS "Performance Rating"
FROM RecentPerformance rp;

-- Operational metrics view
CREATE OR REPLACE VIEW SalesForecast.OperationalMetrics AS
SELECT 
    s.StoreID,
    s.ProductCategory,
    COUNT(*) AS DaysWithData,
    SUM(s.SalesAmount) AS TotalSales,
    AVG(s.SalesAmount) AS AvgDailySales,
    MIN(s.SalesAmount) AS MinDailySales,
    MAX(s.SalesAmount) AS MaxDailySales,
    STDDEV(s.SalesAmount) AS SalesVolatility,
    
    -- Forecasting metrics
    COUNT(p.ID) AS ForecastCount,
    AVG(CASE WHEN p.PredictedSales > 0 THEN 
        ABS((s.SalesAmount - p.PredictedSales) / s.SalesAmount) * 100 
        ELSE NULL END) AS MAPE,
    AVG(s.SalesAmount - p.PredictedSales) AS AvgForecastBias,
    
    -- External factor correlations
    CORR(s.SalesAmount, e.Temperature) AS TempCorrelation,
    CORR(s.SalesAmount, CASE WHEN e.PromotionActive THEN 1 ELSE 0 END) AS PromoCorrelation,
    
    -- Trend indicators
    (SELECT SLOPE FROM (
        SELECT regr_slope(SalesAmount, ROW_NUMBER() OVER (ORDER BY SalesDate)) AS SLOPE
        FROM SalesForecast.SalesData s2
        WHERE s2.StoreID = s.StoreID 
        AND s2.ProductCategory = s.ProductCategory
        AND s2.SalesDate >= CURRENT_DATE - 90
    ) trend) AS TrendSlope,
    
    CURRENT_TIMESTAMP AS LastUpdated

FROM SalesForecast.SalesData s
LEFT JOIN SalesForecast.ModelPredictions p ON 
    s.SalesDate = p.ForecastDate
    AND s.StoreID = p.StoreID
    AND s.ProductCategory = p.ProductCategory
LEFT JOIN SalesForecast.ExternalFactors e ON s.SalesDate = e.FactorDate
WHERE s.SalesDate >= CURRENT_DATE - 90
GROUP BY s.StoreID, s.ProductCategory;

-- =============================================================================
-- Scheduling and Automation
-- =============================================================================

-- Create job scheduler simulation (would use actual job scheduler in production)
CREATE OR REPLACE PROCEDURE SalesForecast.ScheduleDailyTasks()
AS $$
BEGIN
    -- Daily forecast generation
    CALL SalesForecast.GenerateBatchForecast(
        CURRENT_DATE + 1,
        CURRENT_DATE + 7,
        NULL, -- All stores
        NULL  -- All categories
    );
    
    -- Daily performance monitoring
    CALL SalesForecast.CheckModelPerformance();
    
    -- Weekly data quality assessment (on Mondays)
    IF EXTRACT(DOW FROM CURRENT_DATE) = 1 THEN
        DECLARE quality_report TEXT;
        BEGIN
            CALL SalesForecast.AssessDataQuality(quality_report);
            -- In production, would send report via email or dashboard
            RAISE NOTICE 'Weekly Quality Report:\n%', quality_report;
        END;
    END IF;
    
    -- Monthly maintenance (on 1st of month)
    IF EXTRACT(DAY FROM CURRENT_DATE) = 1 THEN
        CALL SalesForecast.MaintenanceCleanup();
    END IF;
    
    -- Quarterly archival (on 1st of quarter)
    IF EXTRACT(DAY FROM CURRENT_DATE) = 1 AND EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 4, 7, 10) THEN
        CALL SalesForecast.ArchiveOldData(730); -- 2 years retention
    END IF;
    
    RAISE NOTICE 'Daily scheduled tasks completed successfully';
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Example Usage and Testing
-- =============================================================================

/*
-- Example usage commands:

-- 1. Load sales data from CSV
CALL SalesForecast.LoadSalesDataFromCSV('/path/to/sales_data.csv', 1000, TRUE);

-- 2. Load external factors
CALL SalesForecast.LoadExternalFactors('WEATHER_API', '2024-01-01', '2024-01-31');

-- 3. Assess data quality
DO $$
DECLARE
    report TEXT;
BEGIN
    CALL SalesForecast.AssessDataQuality(report);
    RAISE NOTICE '%', report;
END $$;

-- 4. API endpoint simulation
SELECT SalesForecast.APIScoreEndpoint('{
    "sales_date": "2024-01-15",
    "store_id": "STORE001",
    "product_category": "Electronics",
    "temperature": 22.5,
    "humidity": 65.0,
    "promotion_active": true,
    "promotion_discount": 0.15,
    "is_holiday": false
}');

-- 5. View executive dashboard
SELECT * FROM SalesForecast.ExecutiveDashboard;

-- 6. View operational metrics
SELECT * FROM SalesForecast.OperationalMetrics;

-- 7. Run scheduled tasks
CALL SalesForecast.ScheduleDailyTasks();

-- 8. Archive old data
CALL SalesForecast.ArchiveOldData(365); -- 1 year retention

-- 9. Maintenance cleanup
CALL SalesForecast.MaintenanceCleanup();

*/

PRINT 'Sales Forecasting data integration and deployment scripts created successfully!';
PRINT 'ETL Procedures: LoadSalesDataFromCSV, LoadExternalFactors';
PRINT 'Quality: AssessDataQuality procedure';
PRINT 'API: APIScoreEndpoint for REST integration';
PRINT 'Maintenance: ArchiveOldData, MaintenanceCleanup, ScheduleDailyTasks';
PRINT 'Views: ExecutiveDashboard, OperationalMetrics';