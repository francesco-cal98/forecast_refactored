import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn import svm
from lightgbm import LGBMRegressor

from utils.metrics import MAPE
from utils.plots import visualize_forecast_and_safety_stock, plot_feat_importance
import pandas as pd
import os
from scipy import stats
from utils.helper import compute_temporal_sample_weights


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
        if self.model_name == "svr":
            return svm.SVR(**self.params)
        elif self.model_name == "light_gbm":
            return LGBMRegressor(**self.params)

        else:
            raise NotImplementedError(f"Model {self.model_name} not implemented.")
    

    # … resto della classe …

    def gridsearch(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   param_grid: dict,
                   X_val: pd.DataFrame = None,
                   cv: int = 3,
                   scoring: str = 'neg_mean_absolute_error'):
        """
        param_grid: dict degli iperparametri
        X_val:    se fornito, serve a costruire il PredefinedSplit sulla sola serie target
        cv:       se X_val is None, numero di split per TimeSeriesSplit
        scoring:  metrica sklearn
        """
        if not param_grid:
            print("⚠️ No parameter grid provided. Skipping grid search.")
            return

        print("🔍 Performing GridSearch...")

        # 1) Decido il cv_split
        if X_val is not None:
            # estraggo il codice del target (X_val contiene solo righe del prodotto da validare)
            target_codes = np.unique(X_val["product_id"].values)
            # -1 = training‐only, 0 = validation
            test_fold = np.where(
                np.isin(X_train["product_id"].values, target_codes),
                0,
               -1
            )
            cv_split = PredefinedSplit(test_fold=test_fold)
        else:
            cv_split = TimeSeriesSplit(n_splits=cv)

        # 2) Inizializzo il GridSearch
        gs = GridSearchCV(
            estimator=self.model,
            param_grid=dict(param_grid),
            cv=cv_split,
            scoring=scoring,
            n_jobs=-1
        )

        # 3) Pesi esponenziali (se li usi)
        weights = compute_temporal_sample_weights(y_train, strategy="exponential", gamma=0.08)

        # 4) (Opzionale) dump per debug
        # X_train.to_excel("x_train_debug.xlsx", index=False)
        # y_train.to_excel("y_train_debug.xlsx", index=False)

        # 5) Fit
        gs.fit(X_train, y_train, sample_weight=weights)

        # 6) Salva il best estimator e le feature
        self.model = gs.best_estimator_
        self.regressor_names = getattr(gs, "feature_names_in_", None)
        print(f"✅ Best parameters: {gs.best_params_}")

        


    def predict(self, X_test, y_test=None, error_fun=MAPE):
        y_pred = self.model.predict(X_test)

        X_test.to_excel(r"C:\Users\fraca\forecast_refactored\x_test.xlsx")
        y_test.to_excel(r"C:\Users\fraca\forecast_refactored\y_test.xlsx")

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

        # transform back outputs
        y_train = np.expm1(y_train)
        y_pred_train = np.expm1(y_pred_train)
        y_pred_test = np.expm1(y_pred_test) 
        y_test = np.expm1(y_test)

        # 🔐 Calcolo della safety stock
        forecast_error_std = np.std(y_pred_train[-safety_stock_time:] - y_train[-safety_stock_time:])
        z_score = stats.norm.ppf(expected_service_level)
        safety_stock = z_score * forecast_error_std * np.sqrt(lead_time)
        print(f"🔐 Safety stock calcolata: {safety_stock:.2f}")

        # ⚠️ Calcolo del bias (sovra/sottostima)
        bias_window = 3
        forecast_errors = y_pred_train - y_train
        rolling_bias = pd.Series(forecast_errors).rolling(window=bias_window, min_periods=1).mean()
        last_bias = rolling_bias.iloc[-1]
        print(f"📏 Rolling forecast bias (ultime {bias_window} settimane): {last_bias:.2f}")

        # 📡 Tracking Signal
        MAD = np.mean(np.abs(forecast_errors - np.mean(forecast_errors)))
        tracking_signal = np.sum(forecast_errors) / (MAD + 1e-8)
        print(f"📡 Tracking Signal: {tracking_signal:.2f}")
        if abs(tracking_signal) > 4:
            print("⚠️ Forecast bias eccessivo! Tracking signal sopra soglia.")

        # 📉 Correzione del forecast test
        y_pred_test_corrected = y_pred_test - last_bias
        mape = MAPE(y_test,y_pred_test_corrected)
        print(f"📈 MAPE: {mape:.2f}%")

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
        # Imposto stock iniziale come media finale dello storico
        current_stock = np.mean(stock_train[-weeks_to_cover:])

        stock_test = []
        stock_test1 = []  # 👈 Prima voce: stock iniziale
        forecast_sell_in_test = []
        lost_sales = []
        n_test = len(y_test)

        for i in range(n_test):
            # 🔮 Stima della domanda futura
            future_demand = np.mean(y_pred_test_corrected[i:i + weeks_to_cover])
            required_stock = future_demand + safety_stock

            # 📦 Ordine da fare per arrivare a livello desiderato
            order = max(0, required_stock - current_stock)
            forecast_sell_in_test.append(order)

            # 🛬 Ordine arriva (si assume arrivo immediato)
            current_stock += order

            # 🛍️ Domanda della settimana
            demand = y_test.iloc[i]

            # ❌ Calcolo vendite perse e aggiornamento stock
            if demand > current_stock:
                lost = demand - current_stock
                lost_sales.append(lost)
                current_stock = 0
            else:
                lost_sales.append(0)
                current_stock -= demand

            # 🧾 Salvo lo stock residuo
            stock_test.append(current_stock)

            # 🧮 Stock alternativo: solo ordine + safety - domanda (NON cumulativo)
            stock_test1.append((order + safety_stock) - demand)

        stock_test1[0] = np.mean(stock_train[-weeks_to_cover:])


        # ✅ Calcolo livello di servizio
        total_demand = y_test.sum()
        total_lost = np.array(lost_sales).sum()
        service_level = 100 * total_demand / (total_demand + total_lost + 1e-5)
        print(f"✅ Service Level: {service_level:.2f}%")

        # 📦 Stock cover
        stock_cover = stock_test / (y_pred_test + 1e-5)

        # Visualizzazione
        fig = visualize_forecast_and_safety_stock(product=product_name,
                                                y_pred_prod=y_pred_test_corrected,
                                                y=y_test,
                                                weekly_stock=stock_test,
                                                safety_stock=np.full_like(y_pred_test, safety_stock),
                                                required_sell_in=forecast_sell_in_test,
                                                stock_cover=stock_cover
                                                )
        
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
            "y_pred_test": y_pred_test_corrected,
            "stock_test": stock_test,
            "forecast_sell_in_test": forecast_sell_in_test,
            "lost_sales": lost_sales,
            "safety_stock": safety_stock,
            "service_level": service_level,
            "stock_cover": stock_cover,
        })




