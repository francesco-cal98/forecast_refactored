from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils.helper import compute_temporal_sample_weights
from utils.metrics import WAPE


class RandomForestForecaster:
    """RandomForest wrapper with the same fit/predict interface as Forecaster."""

    def __init__(self, params: Optional[Dict] = None):
        default = dict(
            n_estimators=700,
            max_features=0.5,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        if params:
            default.update(params)
        self.model = RandomForestRegressor(**default)
        self.imputer = SimpleImputer(strategy="median")
        self.regressor_names = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        dates: Optional[pd.Series] = None,
        recency_gamma: float = 0.08,
    ):
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X_train.drop(columns=non_numeric) if non_numeric else X_train
        X_fit = self.imputer.fit_transform(X_num)

        weights = None
        if dates is not None:
            dates = pd.to_datetime(dates, errors="coerce")
            weights = compute_temporal_sample_weights(
                dates, strategy="exponential", gamma=recency_gamma
            )

        self.model.fit(X_fit, y_train, sample_weight=weights)
        self.regressor_names = getattr(
            self.model, "feature_names_in_", X_num.columns.to_numpy()
        )

    def predict(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        error_fun=WAPE,
        *,
        train_on_log: bool = False,
    ):
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X.drop(columns=non_numeric) if non_numeric else X
        X_pred = self.imputer.transform(X_num)
        y_pred = self.model.predict(X_pred)
        if train_on_log:
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y) if y is not None else None
        else:
            y_true = y
        err = error_fun(y_true, y_pred) if (y_true is not None) else None
        return y_pred, err
