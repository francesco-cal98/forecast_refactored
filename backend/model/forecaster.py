# =============================================================================
# forecaster.py -- Model wrapper and stock analysis
#
# HOW IT WORKS:
#   The Forecaster class wraps sklearn/LightGBM models behind a consistent
#   fit/predict interface. Key responsibilities:
#
#   fit()   -- strips non-numeric columns, computes temporal sample weights
#              (exponential decay: recent rows matter more than old ones),
#              and calls model.fit() with those weights.
#
#   predict() -- runs model.predict(), optionally exponentiates if the model
#               was trained on log-transformed targets.
#
#   stock_analysis() -- given train/test predictions, runs a full inventory
#                       simulation: safety stock, bias correction, tracking
#                       signal, stock cover, and service level.
#
# SUPPORTED MODELS (set via config model.name):
#   "random_forest"    -- sklearn RandomForestRegressor
#   "gradient_boosting"-- sklearn GradientBoostingRegressor
#   "light_lgm"        -- LightGBM LGBMRegressor (recommended, Tweedie loss)
#   "svr"              -- sklearn SVR (does not support sample weights)
# =============================================================================

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from scipy import stats

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


class _ETSModelProxy:
    """
    Thin proxy so backtest.py can call fore.model.predict(Xtr_num) on an ETS
    model and get back the in-sample fitted values (same length as X, ignoring X).
    """
    def __init__(self, fitted_values: np.ndarray):
        self._fitted = np.asarray(fitted_values)

    def predict(self, X=None) -> np.ndarray:
        return self._fitted

from utils.metrics import MAPE, WAPE
from utils.helper import compute_temporal_sample_weights
from utils.plots import (
    visualize_forecast_and_safety_stock,
    visualize_forecast_fanchart,
    visualize_inventory_scenarios,
    plot_feat_importance,
)

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
        if self.model_name == "ets":
            # ETS is stateful — model is set during fit(), not here
            return None
        raise NotImplementedError(f"{self.model_name} not implemented")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        dates: Optional[pd.Series] = None,
        recency_gamma: float = 0.08,
        sample_weight: Optional[np.ndarray] = None,
    ):
        # ── ETS path: ignores X, fits Holt-Winters on y_train ──────────────
        if self.model_name == "ets":
            if not _HAS_STATSMODELS:
                raise ImportError("statsmodels is required for ETS. Run: pip install statsmodels")
            y_arr = np.asarray(y_train, dtype=float)
            sp = int(self.params.get("seasonal_periods", 12))
            trend = str(self.params.get("trend", "add"))
            # Use multiplicative seasonal only when we have enough cycles AND all-positive target
            n_cycles = len(y_arr) / sp
            has_zeros = np.any(y_arr <= 0)
            seasonal = "mul" if (n_cycles >= 1.5 and not has_zeros) else "add"
            if seasonal == "add" and str(self.params.get("seasonal", "mul")) == "mul":
                print(f"[ETS] falling back to additive seasonality "
                      f"(cycles={n_cycles:.1f}, has_zeros={has_zeros})")
            hw = ExponentialSmoothing(
                y_arr,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp,
                damped_trend=bool(self.params.get("damped_trend", True)),
                initialization_method="estimated",
            )
            self._ets_result = hw.fit(optimized=True)
            fitted = np.asarray(self._ets_result.fittedvalues)
            # Replace leading NaNs (ETS burn-in) with first valid value
            if np.any(np.isnan(fitted)):
                first_valid = fitted[~np.isnan(fitted)][0] if np.any(~np.isnan(fitted)) else 0.0
                fitted = np.where(np.isnan(fitted), first_valid, fitted)
            self.model = _ETSModelProxy(fitted)
            self.regressor_names = None
            return
        # ── ML path (sklearn / LightGBM) ───────────────────────────────────
        non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        X_num = X_train.drop(columns=non_numeric) if non_numeric else X_train
        weights = sample_weight
        if weights is None and dates is not None:
            dates = pd.to_datetime(dates, errors="coerce")
            # Compute exponential decay weights: recent rows get weight~1, old rows get weight~0
            weights = compute_temporal_sample_weights(dates, strategy="exponential", gamma=recency_gamma)
        try:
            # Fix: pass computed weights instead of hardcoded None
            self.model.fit(X_num, y_train, sample_weight=weights)
        except TypeError:
            # SVR does not support sample_weight — fall back silently
            self.model.fit(X_num, y_train)
            print(f"[Forecaster.fit] sample_weight applied=False (provided={weights is not None})")
        self.regressor_names = getattr(self.model, "feature_names_in_", X_num.columns.to_numpy())

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None, error_fun=WAPE, *, train_on_log: bool = False):
        # ── ETS path: forecast n_steps beyond the training window ───────────
        if self.model_name == "ets":
            n_steps = len(y) if y is not None else len(X)
            y_pred = np.maximum(0.0, self._ets_result.forecast(n_steps))
            y_true = y
            err = error_fun(y_true, y_pred) if (y_true is not None) else None
            return y_pred, err
        # ── ML path ─────────────────────────────────────────────────────────
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
        oof_residuals: Optional[np.ndarray] = None,
    ):
        print("[stock_analysis] Starting inventory simulation...")

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

        # ── Point-by-point empirical safety stock ────────────────────────────
        # Relative residuals: (actual - forecast) / forecast
        #   positive = actual > predicted = unmet demand risk
        #
        # Source priority:
        #   1. OOF residuals from prior backtest cuts  ← out-of-sample, unbiased
        #   2. Recent training residuals × 1.5 inflation ← in-sample fallback
        #      (training errors are optimistic; inflation partially corrects that)
        #
        # SS[t] = rel_buffer × forecast[t] × √(lead_time)
        # → December automatically gets proportionally more safety stock.
        _OOF_MIN_PTS = 3          # minimum OOF points for a reliable quantile
        _TRAIN_INFLATION = 1.5    # inflate training quantile to compensate optimism

        window = min(safety_stock_time, len(y_train))
        _pred_w = y_pred_train[-window:]
        _true_w = y_train[-window:]
        _train_rel = (_true_w - _pred_w) / (np.abs(_pred_w) + 1e-6)

        # Always include inflated training residuals as a seasonal prior.
        # OOF residuals (when available) are appended unweighted — they're unbiased
        # but may cover a different season than the test window (e.g. summer OOF
        # calibrating a winter test).  By pooling both, December training residuals
        # still anchor the quantile even when OOF data is scarce or off-season.
        _rel_for_ss = _train_rel * _TRAIN_INFLATION   # start with inflated training
        if oof_residuals is not None and len(oof_residuals) >= _OOF_MIN_PTS:
            _rel_for_ss = np.concatenate([_rel_for_ss, np.asarray(oof_residuals, dtype=float)])
            _ss_source  = (f"train×{_TRAIN_INFLATION}(n={len(_train_rel)}) "
                           f"+ OOF(n={len(oof_residuals)}, unbiased)")
        else:
            n_oof = len(oof_residuals) if oof_residuals is not None else 0
            _ss_source  = f"train×{_TRAIN_INFLATION} only (OOF n={n_oof} < {_OOF_MIN_PTS})"

        rel_buffer = float(np.quantile(_rel_for_ss, expected_service_level))
        rel_buffer = max(rel_buffer, 0.0)   # SS can't be negative

        # Point-by-point arrays
        safety_stock_test  = rel_buffer * y_pred_test * np.sqrt(lead_time)
        safety_stock_train = rel_buffer * y_pred_train * np.sqrt(lead_time)
        # Scalar summary for printing
        safety_stock = float(np.mean(safety_stock_test))
        print(f"  Safety stock rel_buffer: {rel_buffer:.1%}  [{_ss_source}]  "
              f"mean={safety_stock:.0f}  "
              f"range=[{safety_stock_test.min():.0f}–{safety_stock_test.max():.0f}]")

        # Bias
        bias_window = min(3, len(y_train))
        forecast_errors = y_pred_train - y_train
        rolling_bias = pd.Series(forecast_errors).rolling(window=bias_window, min_periods=1).mean()
        last_bias = float(rolling_bias.iloc[-1])
        print(f"  Rolling bias (last {bias_window} weeks): {last_bias:.2f}")

        # Tracking Signal
        MAD = np.mean(np.abs(forecast_errors - np.mean(forecast_errors))) + 1e-8
        tracking_signal = np.sum(forecast_errors) / MAD
        print(f"  Tracking signal: {tracking_signal:.2f}")
        if abs(tracking_signal) > 4:
            print("  WARNING: tracking signal above threshold (|ts| > 4) — systematic bias detected")

        # Bias correction on test preds
        y_pred_test_corrected = np.maximum(0, y_pred_test - last_bias)
        mape = WAPE(y_test, y_pred_test_corrected)
        print(f"  WAPE (bias-corrected): {mape:.2f}%")

        p50 = y_pred_test_corrected
        p10 = None
        p90 = None
        try:
            residuals_train = y_train - y_pred_train
            if residuals_train.size > 0:
                q10, q90 = np.nanquantile(residuals_train, [0.1, 0.9])
                p10 = np.maximum(0, p50 + q10)
                p90 = np.maximum(0, p50 + q90)
        except Exception:
            pass

        # Dynamic weeks-to-cover
        std_forecast = np.std(y_pred_test_corrected)
        if std_forecast < 100:
            weeks_to_cover = 1
        elif std_forecast < 300:
            weeks_to_cover = 2
        else:
            weeks_to_cover = 3
        print(f"  Weeks-to-cover (dynamic): {weeks_to_cover}")

        # Train simulation
        n_train = len(y_pred_train)
        stock_train = np.zeros(n_train)
        forecast_sell_in_train = np.zeros(n_train)
        current_stock = 0.0

        for i in range(n_train):
            future_demand = float(np.mean(y_pred_train[i:i + weeks_to_cover]))
            required_stock = future_demand + float(safety_stock_train[i])
            order = max(0.0, required_stock - current_stock)
            forecast_sell_in_train[i] = order
            current_stock = max(0.0, current_stock + order - float(y_train[i]))
            stock_train[i] = current_stock

        # Test simulation
        current_stock = float(np.mean(stock_train[max(0, n_train - weeks_to_cover):]))
        initial_test_stock = current_stock
        stock_test = []
        stock_test1 = []
        forecast_sell_in_test = []
        lost_sales = []
        n_test = len(y_test)

        for i in range(n_test):
            future_demand = float(np.mean(y_pred_test_corrected[i:i + weeks_to_cover]))
            required_stock = future_demand + float(safety_stock_test[i])
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
            stock_test1.append((order + float(safety_stock_test[i])) - demand)

        if stock_test:
            stock_test1[0] = float(np.mean(stock_train[max(0, n_train - weeks_to_cover):]))

        demand_low = p10 if p10 is not None else p50
        demand_base = p50
        demand_high = p90 if p90 is not None else p50

        def _simulate_stock_with_fixed_orders(demand_series):
            s = float(initial_test_stock)
            stock_path = []
            lost_path = []
            for i in range(n_test):
                s += float(forecast_sell_in_test[i])
                d = float(demand_series[i])
                if d > s:
                    lost_path.append(d - s)
                    s = 0.0
                else:
                    lost_path.append(0.0)
                    s -= d
                stock_path.append(s)
            return np.asarray(stock_path), np.asarray(lost_path)

        stock_low, lost_sales_low = _simulate_stock_with_fixed_orders(demand_low)
        stock_base, lost_sales_base = _simulate_stock_with_fixed_orders(demand_base)
        stock_high, lost_sales_high = _simulate_stock_with_fixed_orders(demand_high)

        total_demand_low = float(np.sum(demand_low))
        total_demand_base = float(np.sum(demand_base))
        total_demand_high = float(np.sum(demand_high))
        service_level_low = 100.0 * total_demand_low / (total_demand_low + float(np.sum(lost_sales_low)) + 1e-5)
        service_level_base = 100.0 * total_demand_base / (total_demand_base + float(np.sum(lost_sales_base)) + 1e-5)
        service_level_high = 100.0 * total_demand_high / (total_demand_high + float(np.sum(lost_sales_high)) + 1e-5)

        total_demand = float(np.sum(y_test))
        total_lost = float(np.sum(lost_sales))
        service_level = 100.0 * total_demand / (total_demand + total_lost + 1e-5)
        print(f"  Service level: {service_level:.2f}%")

        stock_test_arr = np.asarray(stock_test)
        stock_cover = stock_test_arr / (y_pred_test_corrected + 1e-5)

        fig = visualize_forecast_and_safety_stock(
            product=product_name,
            y_pred_prod=y_pred_test_corrected,
            y=y_test,
            weekly_stock=stock_test_arr,
            safety_stock=safety_stock_test,          # point-by-point array
            required_sell_in=forecast_sell_in_test,
            stock_cover=stock_cover,
        )
        fig_train = visualize_forecast_and_safety_stock(
            product=product_name,
            y_pred_prod=y_pred_train,
            y=y_train,
            weekly_stock=stock_train,
            safety_stock=safety_stock_train,         # point-by-point array
            required_sell_in=forecast_sell_in_train,
            stock_cover=np.zeros_like(y_pred_train),
        )
        fig_fan = visualize_forecast_fanchart(
            product=product_name,
            y_true=y_test,
            p50=p50,
            p10=p10,
            p90=p90,
        )
        fig_inventory_band = visualize_inventory_scenarios(
            product=product_name,
            stock_low=stock_low,
            stock_base=stock_base,
            stock_high=stock_high,
            safety_stock_line=safety_stock_test,  # point-by-point, not flat scalar
        )

        if save_fp:
            os.makedirs(save_fp, exist_ok=True)
            fig.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis.png"), bbox_inches="tight")
            fig_train.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis_train.png"), bbox_inches="tight")
            fig_fan.savefig(os.path.join(save_fp, f"{product_name}_forecast_fanchart.png"), bbox_inches="tight")
            fig_inventory_band.savefig(os.path.join(save_fp, f"{product_name}_inventory_scenario_band.png"), bbox_inches="tight")

        if hasattr(self.model, "feature_importances_"):
            try:
                feat_imp_path = (
                    os.path.join(save_fp, f"{product_name}_feature_importance.png")
                    if save_fp else None
                )
                plot_feat_importance(
                    importance=self.model.feature_importances_,
                    feature_names=self.regressor_names,
                    save_path=feat_imp_path,
                )
            except Exception as e:
                print(f"  [feature importance] could not save: {e}")

        return pd.DataFrame({
            "y_test": y_test,
            "y_pred_test": y_pred_test_corrected,
            "stock_test": stock_test_arr,
            "forecast_sell_in_test": forecast_sell_in_test,
            "lost_sales": lost_sales,
            "stock_low": stock_low,
            "stock_base": stock_base,
            "stock_high": stock_high,
            "lost_sales_low": lost_sales_low,
            "lost_sales_base": lost_sales_base,
            "lost_sales_high": lost_sales_high,
            "safety_stock": safety_stock_test,       # point-by-point array
            "service_level": service_level,
            "service_level_low": service_level_low,
            "service_level_base": service_level_base,
            "service_level_high": service_level_high,
            "stock_cover": stock_cover,
        })
