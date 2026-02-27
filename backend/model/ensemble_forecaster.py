from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class EnsembleForecaster:
    """Ridge meta-learner on top of base model predictions."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.meta = Ridge(alpha=self.alpha)
        self.is_fitted = False

    def fit(self, pred_lgbm: np.ndarray, pred_rf: np.ndarray, y_true: np.ndarray):
        x1 = np.asarray(pred_lgbm, dtype=float)
        x2 = np.asarray(pred_rf, dtype=float)
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
        if mask.sum() < 8:
            self.is_fitted = False
            return
        x = np.column_stack([x1[mask], x2[mask]])
        self.meta.fit(x, y[mask])
        self.is_fitted = True

    def predict(self, pred_lgbm: np.ndarray, pred_rf: np.ndarray) -> np.ndarray:
        x1 = np.asarray(pred_lgbm, dtype=float)
        x2 = np.asarray(pred_rf, dtype=float)
        x = np.column_stack([x1, x2])
        if not self.is_fitted:
            # Fallback: simple average if meta has not enough training points.
            return 0.5 * x1 + 0.5 * x2
        return self.meta.predict(x)

    @property
    def coef_(self) -> np.ndarray:
        if not self.is_fitted:
            return np.array([0.5, 0.5], dtype=float)
        return np.asarray(self.meta.coef_, dtype=float)
