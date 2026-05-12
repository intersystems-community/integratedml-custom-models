-- ----------------------------------------------------------------------------
-- TabPFN-3 demo — evaluation queries
--
-- Holds the model under tests with accuracy / MAE / RMSE-style aggregates so
-- you can compare TabPFN-3 against alternative IntegratedML models in the
-- same notebook without leaving SQL.
-- ----------------------------------------------------------------------------

-- ----- Classification metrics -----
WITH pred AS (
    SELECT
        patient_id,
        needs_followup AS actual,
        PREDICT(PatientRiskScreener) AS predicted
    FROM TabPFN.PatientScreening
    WHERE split = 'test'
)
SELECT
    COUNT(*)                                                   AS n,
    AVG(CASE WHEN actual = predicted THEN 1.0 ELSE 0.0 END)   AS accuracy,
    SUM(CASE WHEN actual = 1 AND predicted = 1 THEN 1 ELSE 0 END) AS true_positive,
    SUM(CASE WHEN actual = 0 AND predicted = 1 THEN 1 ELSE 0 END) AS false_positive,
    SUM(CASE WHEN actual = 1 AND predicted = 0 THEN 1 ELSE 0 END) AS false_negative,
    SUM(CASE WHEN actual = 0 AND predicted = 0 THEN 1 ELSE 0 END) AS true_negative
FROM pred;

-- Hardest cases — actual positives the model missed.
SELECT patient_id, age, sex, bmi, systolic_bp, fasting_glucose, hdl, ldl,
       smoker, family_history
FROM TabPFN.PatientScreening
WHERE split = 'test'
  AND needs_followup = 1
  AND PREDICT(PatientRiskScreener) = 0;


-- ----- Regression metrics -----
WITH pred AS (
    SELECT
        building_id,
        kwh_day AS actual,
        PREDICT(BuildingEnergyForecaster) AS predicted
    FROM TabPFN.BuildingEnergy
    WHERE split = 'test'
)
SELECT
    COUNT(*)                                       AS n,
    AVG(ABS(actual - predicted))                   AS mae,
    SQRT(AVG(POWER(actual - predicted, 2)))        AS rmse,
    AVG(ABS(actual - predicted) / NULLIF(actual, 0)) AS mape
FROM pred;

-- Largest absolute errors — useful for inspecting outliers / model failures.
SELECT
    building_id, floor_area, occupants, hvac_type, outdoor_temp_c,
    insulation_rating, kwh_day AS actual,
    PREDICT(BuildingEnergyForecaster) AS predicted,
    ABS(kwh_day - PREDICT(BuildingEnergyForecaster)) AS abs_error
FROM TabPFN.BuildingEnergy
WHERE split = 'test'
ORDER BY ABS(kwh_day - PREDICT(BuildingEnergyForecaster)) DESC
FETCH FIRST 10 ROWS ONLY;
