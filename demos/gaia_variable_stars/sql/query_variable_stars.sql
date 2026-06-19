-- Query variable stars above threshold X%
-- Replace :threshold with desired percentage (e.g. 10 for 10%)

SELECT
    s.source_id,
    s.ra,
    s.dec,
    s.min_mag                                          AS phot_g_mean_mag_min,
    s.max_mag                                          AS phot_g_mean_mag_max,
    ROUND((s.mag_range / s.mean_mag) * 100.0, 4)      AS pct_change,
    PREDICT(GaiaVariability USING s.*) AS ml_is_variable
FROM GaiaObservationStats s
WHERE
    ROUND((s.mag_range / s.mean_mag) * 100.0, 4) >= :threshold
ORDER BY pct_change DESC;
