# usdcop_forecasting_clean/backend/src/features/transformer.py
"""
Feature transformation module.

Handles scaling and other transformations.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """
    Transforms features with various scaling methods.

    Follows Strategy pattern - can swap scaling strategies.
    """

    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }

    def __init__(self, method: str = 'standard'):
        if method not in self.SCALERS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.SCALERS.keys())}")

        self.method = method
        self.scaler = self.SCALERS[method]()
        self._is_fitted = False
        self.feature_names: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit scaler on training data.

        Args:
            X: Training features

        Returns:
            self for method chaining
        """
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()
        self._is_fitted = True

        logger.info(f"Fitted {self.method} scaler on {len(self.feature_names)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.

        Args:
            X: Features to transform

        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform to original scale.

        Args:
            X: Scaled features

        Returns:
            Features in original scale
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted.")

        X_original = pd.DataFrame(
            self.scaler.inverse_transform(X),
            index=X.index,
            columns=X.columns
        )

        return X_original

    def save(self, path: Path):
        """Save transformer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'scaler': self.scaler,
            'method': self.method,
            'feature_names': self.feature_names,
            'is_fitted': self._is_fitted
        }, path)

        logger.info(f"Saved transformer to {path}")

    @classmethod
    def load(cls, path: Path) -> 'FeatureTransformer':
        """Load transformer from disk."""
        data = joblib.load(path)

        transformer = cls(method=data['method'])
        transformer.scaler = data['scaler']
        transformer.feature_names = data['feature_names']
        transformer._is_fitted = data['is_fitted']

        return transformer

    def get_stats(self) -> dict:
        """Get scaling statistics."""
        if not self._is_fitted:
            return {}

        if self.method == 'standard':
            return {
                'mean': dict(zip(self.feature_names, self.scaler.mean_)),
                'std': dict(zip(self.feature_names, self.scaler.scale_))
            }
        elif self.method == 'minmax':
            return {
                'min': dict(zip(self.feature_names, self.scaler.data_min_)),
                'max': dict(zip(self.feature_names, self.scaler.data_max_))
            }
        elif self.method == 'robust':
            return {
                'center': dict(zip(self.feature_names, self.scaler.center_)),
                'scale': dict(zip(self.feature_names, self.scaler.scale_))
            }

        return {}
