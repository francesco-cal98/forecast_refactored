import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit

from utils.metrics import MAPE
from utils.plots import visualize_forecast_and_safety_stock, plot_feat_importance
import pandas as pd
import os
from scipy import stats


class Forecaster:
    def __init__(self, model_name="RandomForestRegressor", params=None):
        self.model_name = model_name
        self.params = params if params is not None else {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        # Per ora solo RandomForest, ma puoi estendere facilmente
        if self.model_name == "random_forest":
            return RandomForestRegressor(**self.params)
        if self.model_name == "gradient_boosting":
            return GradientBoostingRegressor(**self.params)
        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")
    
    def gridsearch(self, X_train, y_train, param_grid=None, cv=5, scoring='neg_mean_squared_error'):
        if not param_grid:
            print("⚠️ No parameter grid provided. Skipping grid search.")
            return

        print("🔍 Performing GridSearch...")
        gs = GridSearchCV(self.model, dict(param_grid), cv=TimeSeriesSplit(n_splits=cv).split(X_train), scoring=scoring, n_jobs=-1)
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        self.regressor_names = gs.feature_names_in_
        print(f"✅ Best parameters: {gs.best_params_}")


    def predict(self, X_test, y_test=None, error_fun=MAPE):
        y_pred = self.model.predict(X_test)

        if y_test is not None:
            error = error_fun(y_test, y_pred)
        else:
            error = None
        return y_pred, error
    
    def stock_analysis(self,
                    y_test: pd.Series,
                    y_pred_test: pd.Series,
                    y_train: pd.Series,
                    y_pred_train: pd.Series,
                    sell_in_true: pd.Series = None,
                    safety_stock_time: int = 10,
                    lead_time: int = 1,
                    expected_service_level: float = 0.95,
                    product_name: str = "Prodotto",
                    save_fp: str = "outputs"):
        """
        Simula una gestione dello stock sulla base delle previsioni e visualizza i risultati.
        """

        print("📦 Avvio analisi dello stock...")

        # 🔐 Calcolo della safety stock
        forecast_error_std = np.std(y_pred_train[-safety_stock_time:] - y_train[-safety_stock_time:])
        z_score = stats.norm.ppf(expected_service_level)
        safety_stock = z_score * forecast_error_std * np.sqrt(lead_time)
        print(f"🔐 Safety stock calcolata: {safety_stock:.2f}")

        # ⚠️ Calcolo del bias (sovra/sottostima)
        forecast_bias = np.mean(y_pred_train - y_train)
        print(f"📏 Forecast bias medio: {forecast_bias:.2f}")

        # 📉 Correzione del bias nelle previsioni test
        y_pred_test_corrected = y_pred_test - forecast_bias

        # 🔄 weeks_to_cover dinamico (più varianza => più copertura)
        std_forecast = np.std(y_pred_test)
        if std_forecast < 100:
            weeks_to_cover = 1
        elif std_forecast < 300:
            weeks_to_cover = 2
        else:
            weeks_to_cover = 3
        print(f"🗓️ Weeks to cover dinamico impostato a: {weeks_to_cover}")

        # ▶️ SIMULAZIONE STOCK SUL TRAINING
        n_train = len(y_pred_train)
        stock_train = np.zeros(n_train)
        forecast_sell_in_train = np.zeros(n_train)
        current_stock = 0.0
        y_train = np.array(y_train)

        for i in range(n_train):
            future_demand = np.mean(y_pred_train[i:i + weeks_to_cover])
            required_stock = future_demand + safety_stock

            forecast_sell_in_train[i] = max(0, required_stock - current_stock)
            current_stock = max(0, current_stock + forecast_sell_in_train[i] - y_train[i])
            stock_train[i] = current_stock

        # 🟢 IMPOSTO LO STOCK INIZIALE PER IL TESTING
        current_stock = np.mean(stock_train[-weeks_to_cover:])

        # ▶️ SIMULAZIONE STOCK SUL TEST
        n_test = len(y_pred_test)
        stock_test = np.zeros(n_test)
        forecast_sell_in_test = np.zeros(n_test)
        lost_sales = np.zeros(n_test)

        for i in range(n_test):
            future_demand = np.mean(y_pred_test_corrected[i:i + weeks_to_cover])
            required_stock = future_demand + safety_stock

            forecast_sell_in_test[i] = max(0, required_stock - current_stock)
            current_stock = max(0, current_stock + forecast_sell_in_test[i] - y_test[i])
            stock_test[i] = current_stock

            available = stock_test[i] + forecast_sell_in_test[i]
            if available < y_test[i]:
                lost_sales[i] = y_test[i] - available

        # ✅ Calcolo livello di servizio
        total_demand = y_test.sum()
        total_lost = lost_sales.sum()
        service_level = 100 * total_demand / (total_demand + total_lost + 1e-5)
        print(f"✅ Service Level: {service_level:.2f}%")

        # 📦 Stock cover
        stock_cover = stock_test / (y_pred_test + 1e-5)

        # Visualizzazione
        fig = visualize_forecast_and_safety_stock(product=product_name,
                                                y_pred_prod=y_pred_test,
                                                y=y_test,
                                                weekly_stock=stock_test,
                                                safety_stock=np.full_like(y_pred_test, safety_stock),
                                                required_sell_in=forecast_sell_in_test,
                                                stock_cover=stock_cover)
        
        fig_train = visualize_forecast_and_safety_stock(product=product_name,
                                                y_pred_prod=y_pred_train,
                                                y=y_train,
                                                weekly_stock=stock_train,
                                                safety_stock=np.full_like(y_pred_train, safety_stock),
                                                required_sell_in=forecast_sell_in_train,
                                                stock_cover=np.zeros_like(y_pred_train))
        
        
        # TO BE REMOVED
        MAPE_value = MAPE(y_test, forecast_sell_in_test)
        print(f"📈 MAPE: {MAPE_value:.2f}%")
        # -- Feature Importance (opzionale) --
        if hasattr(self.model, 'feature_importances_'):
            feature_importance_fig = plot_feat_importance(importance=self.model.feature_importances_,
                                feature_names=self.regressor_names)
            feat_impo_df = pd.DataFrame({
                'Feature': self.regressor_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

        
        if save_fp:
            fig.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis.png"))
            fig_train.savefig(os.path.join(save_fp, f"{product_name}_stock_analysis_train.png"))
            #eature_importance_fig.savefig(os.path.join(save_fp, f"{product_name}_feature_importance.png"))

        return pd.DataFrame({
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "stock_test": stock_test,
            "forecast_sell_in_test": forecast_sell_in_test,
            "lost_sales": lost_sales,
            "safety_stock": safety_stock,
            "service_level": service_level,
            "stock_cover": stock_cover,
        })


