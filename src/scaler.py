'''
Custom scaler for COI indicators
'''
from sklearn.base import BaseEstimator, TransformerMixin


class COIScaler(BaseEstimator, TransformerMixin):
    '''
    Scales values based on input min and max
    '''
    def __init__(self, min_val=1, max_val=100):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, X):
        return X * (self.max_val - self.min_val) + self.min_val