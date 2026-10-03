-- ----------------------------------------------------------------------------
-- Kumo Tabular demo — evaluation queries
--
-- Holds the model under tests with accuracy / MAE / RMSE-style aggregates so
-- you can compare Kumo Tabular against alternative IntegratedML models in the
-- same notebook without leaving SQL.
-- ----------------------------------------------------------------------------

-- ----- Classification metrics -----
WITH pred AS (
    SELECT
        patient_id,
        needs_followup AS actual,
        PREDICT(KumoPatientRiskScreener) AS predicted
    FROM KumoTab.PatientScreening
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
FROM KumoTab.PatientScreening
WHERE split = 'test'
  AND needs_followup = 1
  AND PREDICT(KumoPatientRiskScreener) = 0;


-- ----- Regression metrics -----
WITH pred AS (
    SELECT
        building_id,
        kwh_day AS actual,
        PREDICT(KumoBuildingEnergyForecaster) AS predicted
    FROM KumoTab.BuildingEnergy
    WHERE split = 'test'
)
SELECT
    COUNT(*)                                       AS n,
    AVG(ABS(actual - predicted))                   AS mae,
    SQRT(AVG(POWER(actual - predicted, 2)))        AS rmse,
    AVG(ABS(actual - predicted) / NULLIF(actual, 0)) AS mape
FROM pred;

-- Largest absolute errors — useful for inspecting outliers / model failures.
-- (IRIS 2026.1 fails to compile ORDER BY on a PREDICT() expression combined
-- with FETCH FIRST, so sort a derived table with TOP instead.)
SELECT TOP 10 *
FROM (
    SELECT
        building_id, floor_area, occupants, hvac_type, outdoor_temp_c,
        insulation_rating, kwh_day AS actual,
        PREDICT(KumoBuildingEnergyForecaster) AS predicted,
        ABS(kwh_day - PREDICT(KumoBuildingEnergyForecaster)) AS abs_error
    FROM KumoTab.BuildingEnergy
    WHERE split = 'test'
) errors
ORDER BY abs_error DESC;
