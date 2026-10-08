"""Preprocessing components shared by the CRSS models.

Kept in an importable module (not a script) so saved models can be loaded anywhere.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MedianImputer32(BaseEstimator, TransformerMixin):
    """Median imputation in float32, column by column.

    Equivalent to SimpleImputer(strategy="median"), which needed about 10 GB of temporary
    float64 memory on the full 1,300-column training matrix.
    """

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        A = np.asarray(X, dtype=np.float32)
        med = np.empty(A.shape[1], dtype=np.float32)
        for j in range(A.shape[1]):
            col = A[:, j]
            col = col[~np.isnan(col)]
            med[j] = np.median(col) if col.size else 0.0
        self.statistics_ = med
        return self

    def transform(self, X):
        A = np.array(X, dtype=np.float32)  # copy, so the caller's data is untouched
        rows, cols = np.where(np.isnan(A))
        A[rows, cols] = self.statistics_[cols]
        return A

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)
