from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

_X = np.array([[50, 40, 30, 5, 7, 10, 9, np.nan, 12],
               [1.68, 1.83, 1.77, np.nan, 1.9, 1.65, 1.88, np.nan, 1.75]])
X = np.transpose(_X)

pipe = Pipeline([
    ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('normalize', MinMaxScaler())
])

X_pipe = pipe.fit_transform(X)
X_impute = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
X_normalize = MinMaxScaler().fit_transform(X_impute)
print(np.array_equal(X_pipe, X_normalize))

data = {'IQ': ['high', 'avg', 'avg', 'low', 'high', 'avg', 'high', 'high', None],
        'temper': ['good', None, 'good', 'bad', 'bad', 'bad', 'bad', None, 'bad'],
        'income': [50, 40, 30, 5, 7, 10, 9, np.nan, 12],
        'height': [1.68, 1.83, 1.77, np.nan, 1.9, 1.65, 1.88, np.nan, 1.75]}

X = pd.DataFrame(data)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):
        self.attr_names = attr_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attr_names].values


categorical_features = ['IQ', 'temper']
numeric_features = ['income', 'height']

categorical_pipe = Pipeline([
    ('select', DataFrameSelector(categorical_features)),
    ('impute', SimpleImputer(missing_values=None, strategy='most_frequent')),
    ('one_hot_encoder', OneHotEncoder(sparse=False))
])

numeric_pipe = Pipeline([
    ('select', DataFrameSelector(numeric_features)),
    ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('normalize', MinMaxScaler())
])

full_pipe = FeatureUnion(transformer_list=[
    ('numeric_pipe', numeric_pipe),
    ('categorical_pipe', categorical_pipe)
])

X_full = full_pipe.fit_transform(X)
print(X_full)

