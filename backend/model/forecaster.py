import os
import json
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Dict

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import svm

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from utils.metrics import MAPE,WAPE
from utils.plots import visualize_forecast_and_safety_stock, plot_feat_importance
from utils.helper import compute_temporal_sample_weights, parse_time_string
from scipy import stats


import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

from utils.metrics import MAPE
from utils.helper import compute_temporal_sample_weights
from utils.plots import visualize_forecast_and_safety_stock, plot_feat_importance
from scipy import stats
import os

class Forecaster:
    def __init__(self, model_name: str = "random_forest", params: Optional[Dict] = None):
        self.model_name = (model_name or "").lower()
        self.params = params or {}
        self.model = self._initialize_model()
        self.regressor_names = None

    def _initialize_model(self):
        if self.model_name == "random_forest":
            default = dict(n_estimators=900, max_features=0.4, min_samples_leaf=10, random_state=42, n_jobs=-1)
            default.update(self.params); return RandomForestRegressor(**default)
        if self.model_name == "gradient_boosting":
            default = dict(n_estimators=800, learning_rate=0.08, max_depth=3, subsample=1.0)
            default.update(self.params); return GradientBoostingRegressor(**default)
        if self.model_name in ("light_gbm","lgbm","lightgbm") and _HAS_LGBM:
            default = dict(n_estimators=800, num_leaves=31, feature_fraction=0.8, min_data_in_leaf=30, learning_rate=0.08)
            default.update(self.params); return LGBMRegressor(**default)
        if self.model_name == "svr":
            return svm.SVR(**self.params)
        raise NotImplementedError(f"{self.model_name} not implemented")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, *, dates: Optional[pd.Series] = None, recency_gamma: float = 0.08):
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X_train.drop(columns=non_numeric) if non_numeric else X_train
        weights = None
        if dates is not None:
            dates = pd.to_datetime(dates, errors="coerce")
            weights = compute_temporal_sample_weights(dates, strategy="exponential", gamma=recency_gamma)
        try:
            self.model.fit(X_num, y_train, sample_weight=None)
        except TypeError:
            self.model.fit(X_num, y_train)
        self.regressor_names = getattr(self.model, "feature_names_in_", X_num.columns.to_numpy())

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None, error_fun=WAPE, *, train_on_log: bool = False):
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X.drop(columns=non_numeric) if non_numeric else X
        y_pred = self.model.predict(X_num)
        if train_on_log:
            y_pred = np.expm1(y_pred); y_true = np.expm1(y) if y is not None else None
        else:
            y_true = y
        err = error_fun(y_true, y_pred) if (y_true is not None) else None
        return y_pred, err

    # ------------------------------------------------------------------
    # Stock analysis
    # ------------------------------------------------------------------
    def stock_analysis(
        self,
        y_test: pd.Series,
        y_pred_test: pd.Series,
        y_train: pd.Series,
        y_pred_train: pd.Series,
        sell_in_true: Optional[pd.Series] = None,
        safety_stock_time: int = 10,
        lead_time: int = 1,
        expected_service_level: float = 0.95,
        product_name: str = "Prodotto",
        save_fp: str = "outputs",
        train_on_log: bool = False,
    ):
        print("📦 Avvio analisi dello stock...")

        # bring to level space if needed
        if train_on_log:
            y_pred_train = np.expm1(y_pred_train)
            y_pred_test = np.expm1(y_pred_test)
            y_train = np.expm1(y_train)
            y_test = np.expm1(y_test)

        y_train = np.asarray(y_train)
        y_pred_train = np.asarray(y_pred_train)
        y_test = np.asarray(y_test)
        y_pred_test = np.asarray(y_pred_test)

        # Safety stock based on recent train window
        window = min(safety_stock_time, len(y_train))
        forecast_error_std = np.std(y_pred_train[-window:] - y_train[-window:])
        z_score = stats.norm.ppf(expected_service_level)
        safety_stock = z_score * forecast_error_std * np.sqrt(lead_time)
        print(f"🔐 Safety stock calcolata: {safety_stock:.2f}")

        # Bias
        bias_window = min(3, len(y_train))
        forecast_errors = y_pred_train - y_train
        rolling_bias = pd.Series(forecast_errors).rolling(window=bias_window, min_periods=1).mean()
        last_bias = float(rolling_bias.iloc[-1])
        print(f"📏 Rolling forecast bias (ultime {bias_window} settimane): {last_bias:.2f}")

        # Tracking Signal
        MAD = np.mean(np.abs(forecast_errors - np.mean(forecast_errors))) + 1e-8
        tracking_signal = np.sum(forecast_errors) / MAD
        print(f"📡 Tracking Signal: {tracking_signal:.2f}")
        if abs(tracking_signal) > 4:
            print("⚠️ Forecast bias eccessivo! Tracking signal sopra soglia.")

        # Bias correction on test preds
        y_pred_test_corrected = y_pred_test - last_bias
        mape = WAPE(y_test, y_pred_test_corrected)
        print(f"📈 MAPE domanda (corretta): {mape:.2f}%")

        # Dynamic weeks-to-cover
        std_forecast = np.std(y_pred_test_corrected)
        if std_forecast < 100:
            weeks_to_cover = 1
        elif std_forecast < 300:
            weeks_to_cover = 2
        else:
            weeks_to_cover = 3
        print(f"🗓️ Weeks to cover dinamico impostato a: {weeks_to_cover}")

        # Train simulation
        n_train = len(y_pred_train)
        stock_train = np.zeros(n_train)
        forecast_sell_in_train = np.zeros(n_train)
        current_stock = 0.0

        for i in range(n_train):
            future_demand = float(np.mean(y_pred_train[i:i + weeks_to_cover]))
            required_stock = future_demand + safety_stock
            order = max(0.0, required_stock - current_stock)
            forecast_sell_in_train[i] = order
            current_stock = max(0.0, current_stock + order - float(y_train[i]))
            stock_train[i] = current_stock

        # Test simulation
        current_stock = float(np.mean(stock_train[max(0, n_train - weeks_to_cover):]))
        stock_test = []
        stock_test1 = []
        forecast_sell_in_test = []
        lost_sales = []
        n_test = len(y_test)

        for i in range(n_test):
            future_demand = float(np.mean(y_pred_test_corrected[i:i + weeks_to_cover]))
            required_stock = future_demand + safety_stock
            order = max(0.0, required_stock - current_stock)
            forecast_sell_in_test.append(order)
            current_stock += order
            demand = float(y_test[i])
            if demand > current_stock:
                lost = demand - current_stock
                lost_sales.append(lost)
                current_stock = 0.0
            else:
                lost_sales.append(0.0)
                current_stock -= demand
            stock_test.append(current_stock)
            stock_test1.append((order + safety_stock) - demand)

        if stock_test:
            stock_test1[0] = float(np.mean(stock_train[max(0, n_train - weeks_to_cover):]))

        total_demand = float(np.sum(y_test))
        total_lost = float(np.sum(lost_sales))
        service_level = 100.0 * total_demand / (total_demand + total_lost + 1e-5)
        print(f"✅ Service Level: {service_level:.2f}%")

        stock_test_arr = np.asarray(stock_test)
        stock_cover = stock_test_arr / (y_pred_test_corrected + 1e-5)

        fig = visualize_forecast_and_safety_stock(
            product=product_name,
            y_pred_prod=y_pred_test_corrected,
            y=y_test,
            weekly_stock=stock_test_arr,
            safety_stock=np.full_like(y_pred_test_corrected, safety_stock),
            required_sell_in=forecast_sell_in_test,
            stock_cover=stock_cover,
        )
        fig_train = visualize_forecast_and_safety_stock(
            product=product_name,
            y_pred_prod=y_pred_train,
            y=y_train,
            weekly_stock=stock_train,
            safety_stock=np.full_like(y_pred_train, safety_stock),
            required_sell_in=forecast_sell_in_train,
            stock_cover=np.zeros_like(y_pred_train),
        )

        if save_fp:
            os.makedirs(save_fp, exist_ok=True)
            fig.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis.png"), bbox_inches="tight")
            fig_train.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis_train.png"), bbox_inches="tight")

        if hasattr(self.model, "feature_importances_"):
            try:
                plot_feat_importance(
                    importance=self.model.feature_importances_,
                    feature_names=self.regressor_names,
                )
            except Exception:
                pass

        return pd.DataFrame({
            "y_test": y_test,
            "y_pred_test": y_pred_test_corrected,
            "stock_test": stock_test_arr,
            "forecast_sell_in_test": forecast_sell_in_test,
            "lost_sales": lost_sales,
            "safety_stock": safety_stock,
            "service_level": service_level,
            "stock_cover": stock_cover,
        })
