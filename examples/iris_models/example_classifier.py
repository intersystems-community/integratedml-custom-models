"""
Minimal IRISModel example — classifier.

Deploy this file to the pathtoclassifiers directory and IRIS will load it
automatically during TRAIN MODEL. The class MUST be named IRISModel.

SQL to use this model:
    CREATE MODEL MyModel PREDICTING (label) FROM MyTable
    USING {"pathtoclassifiers": "/path/to/this/directory", "iscmodelsdisabled": 1}
    TRAIN MODEL MyModel
"""

from sklearn.linear_model import LogisticRegression


class IRISModel:
    name = "example_logistic_regression"

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
