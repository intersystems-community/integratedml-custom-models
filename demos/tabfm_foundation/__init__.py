"""TabFM foundation-model demo for IRIS IntegratedML Custom Models.

Wraps Google Research's TabFM zero-shot tabular foundation model as two
self-contained IRISModel classes — one classifier, one regressor — so that
small-to-medium tabular ML problems can be solved directly from IRIS SQL via
CREATE MODEL / TRAIN MODEL / PREDICT(). Falls back to a scikit-learn model
when the `tabfm` package is unavailable in the IRIS Python environment.
"""
