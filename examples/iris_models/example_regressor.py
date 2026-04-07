"""
Minimal IRISModel example — regressor.

Deploy this file to the pathtoregressors directory and IRIS will load it
automatically during TRAIN MODEL. The class MUST be named IRISModel.

SQL to use this model:
    CREATE MODEL MyModel PREDICTING (amount) FROM MyTable
    USING {"pathtoregressors": "/path/to/this/directory", "iscmodelsdisabled": 1}
    TRAIN MODEL MyModel
"""

from sklearn.linear_model import LinearRegression


class IRISModel:
    name = "example_linear_regression"

    def __init__(self):
        self.model = LinearRegression()

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
