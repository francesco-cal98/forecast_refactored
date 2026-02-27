from __future__ import annotations

import numpy as np


class CrostonSBAForecaster:
    """Croston-SBA forecaster for intermittent demand.

    SBA correction:
      forecast = (1 - alpha / 2) * (smoothed_size / smoothed_interval)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        self.fitted_: np.ndarray | None = None
        self.forecast_: float = 0.0

    def fit(self, y: np.ndarray) -> "CrostonSBAForecaster":
        arr = np.asarray(y, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.maximum(arr, 0.0)

        n = arr.size
        if n == 0:
            self.fitted_ = np.zeros(0, dtype=float)
            self.forecast_ = 0.0
            return self

        nonzero_idx = np.flatnonzero(arr > 0.0)
        if nonzero_idx.size == 0:
            self.fitted_ = np.zeros(n, dtype=float)
            self.forecast_ = 0.0
            return self

        first = int(nonzero_idx[0])
        z = float(arr[first])   # smoothed demand size
        p = 1.0                 # smoothed inter-arrival interval
        q = 1                   # current interval counter
        f = (1.0 - self.alpha / 2.0) * z / max(p, 1e-8)

        fitted = np.zeros(n, dtype=float)
        for t in range(n):
            fitted[t] = max(0.0, f)
            if arr[t] > 0.0:
                z = z + self.alpha * (arr[t] - z)
                p = p + self.alpha * (q - p)
                f = (1.0 - self.alpha / 2.0) * z / max(p, 1e-8)
                q = 1
            else:
                q += 1

        self.fitted_ = fitted
        self.forecast_ = float(max(0.0, f))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        h = int(horizon)
        if h <= 0:
            return np.zeros(0, dtype=float)
        return np.full(h, self.forecast_, dtype=float)

    def predict_in_sample(self) -> np.ndarray:
        if self.fitted_ is None:
            return np.zeros(0, dtype=float)
        return self.fitted_.copy()
