# backend/backtest.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import mlflow

from utils.helper import resolve_config_refs,resolve_target_keys
from utils.metrics import MAPE,WAPE,MASE,SMAPE
from model.croston import CrostonSBAForecaster
from model.rf_forecaster import RandomForestForecaster
from model.ensemble_forecaster import EnsembleForecaster


@dataclass
class BacktestCut:
    boundary: pd.Timestamp
    test_len: pd.Timedelta

@dataclass
class BacktestResult:
    item: str
    cut_idx: int
    boundary: str
    metrics: Dict[str, float]
    y_true: List[float]
    y_pred: List[float]
    cut_dir: str

class BacktestRunner:
    def __init__(
        self,
        dataset_cls,           # SalesDataset
        forecaster_cls,        # Forecaster
        cfg: Any,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "forecast_backtest",
        output_dir: str = "outputs/backtest",
        n_cuts: int = 3,
    ):
        self.dataset_cls = dataset_cls
        self.forecaster_cls = forecaster_cls
        self.cfg = cfg
        self.output_dir = output_dir
        self.n_cuts = n_cuts

        os.makedirs(self.output_dir, exist_ok=True)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # ---------- utilities ----------
    @staticmethod
    def _parse_td(s: str) -> pd.Timedelta:
        s = str(s).strip().lower()
        if s.endswith("w"):
            return pd.Timedelta(weeks=int(s[:-1]))
        if s.endswith("m"):
            return pd.Timedelta(days=30 * int(s[:-1]))
        if s.endswith("d"):
            return pd.Timedelta(days=int(s[:-1]))
        return pd.to_timedelta(s)

    def _build_cuts(self, target_dates: pd.Series, test_len_str: str, n_cuts: int) -> List[BacktestCut]:
        test_len = self._parse_td(test_len_str)
        t_max = pd.to_datetime(target_dates.max())
        cuts: List[BacktestCut] = []
        # es. 3 cut: T-3*len, T-2*len, T-1*len
        for i in range(n_cuts, 0, -1):
            boundary = t_max - i * test_len
            cuts.append(BacktestCut(boundary=boundary, test_len=test_len))
        return cuts

    def _get_test_dates(self, ds) -> pd.Series:
        """
        Prova a recuperare le date del test:
        1) se il dataset espone `test_dates`, usa quello;
        2) altrimenti se la colonna raw della data è in X_test, usa quella;
        3) fallback: range numerico.
        """
        if hasattr(ds, "test_dates") and ds.test_dates is not None:
            return pd.to_datetime(ds.test_dates)
        if hasattr(ds, "date_column") and ds.date_column in ds.X_test.columns:
            return pd.to_datetime(ds.X_test[ds.date_column])
        return pd.Series(range(len(ds.y_test)), name="t")

    @staticmethod
    def _residual_quantiles(residuals: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(residuals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"q10": 0.0, "q50": 0.0, "q90": 0.0}
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        return {"q10": float(q10), "q50": float(q50), "q90": float(q90)}

    # ---------- main API ----------
    def run_for_item(self, item: str) -> Dict[str, Any]:
        # 1) Sonda le date disponibili per costruire i cut

        prepare_config = resolve_target_keys(self.cfg)

        # 1) Dataset
        dataset_cfg = getattr(self.cfg, "data", {})
        dataset_extra_cfg = getattr(self.cfg, "dataset_config", {})  # opzionale nel tuo YAML


        ds_probe = self.dataset_cls(
            csv_file=dataset_cfg.file_path,
            separator=dataset_cfg.separator,
            date_column=dataset_cfg.date_column,
            target_column=dataset_cfg.target_column,
            nr_horizon=dataset_cfg.nr_horizon,
            prepare_config=prepare_config,
            column_selection=self.cfg.column_selection,
            product_var=dataset_cfg.product_column,
            granularity=dataset_cfg.granularity,
            product_name=item,
            config=dataset_extra_cfg
        )
        df_full = ds_probe._prepare_dataset().sort_values([ds_probe.product_var, ds_probe.date_column])
        df_full = ds_probe._encode_time(df_full, time_unit=ds_probe.granularity)
        tgt_dates = df_full.loc[df_full[ds_probe.product_var] == item, ds_probe.date_column]
        cuts = self._build_cuts(tgt_dates, self.cfg.train.test_len, self.n_cuts)

        # Artefatti complessivi per questo item
        results: List[BacktestResult] = []
        artifacts_rows = []
        oof_rows = []
        residual_history: List[float] = []
        meta_history_rows: List[Dict[str, float]] = []
        ensemble_enabled = bool(getattr(self.cfg.train, "ensemble", False))
        ridge_alpha = float(getattr(self.cfg.train, "ridge_alpha", 1.0))

        for cut_idx, cut in enumerate(cuts, start=1):
            # 2) Prepara il dataset per questo cut
            ds = self.dataset_cls(
            csv_file=dataset_cfg.file_path,
            separator=dataset_cfg.separator,
            date_column=dataset_cfg.date_column,
            target_column=dataset_cfg.target_column,
            nr_horizon=dataset_cfg.nr_horizon,
            prepare_config=prepare_config,
            column_selection=self.cfg.column_selection,
            product_var=dataset_cfg.product_column,
            granularity=dataset_cfg.granularity,
            product_name=item,
            config=dataset_extra_cfg
        )
            ds.process_data(
                product=item,
                train_params={
                    "train_len": self.cfg.train.train_len,
                    "test_len": self.cfg.train.test_len,
                    "boundary": str(cut.boundary.date()),
                }
            )

            # 3) Fit + Predict
            fore = self.forecaster_cls(model_name=self.cfg.model.name, params=self.cfg.model.params)

            Xtr_num = ds.X_train.select_dtypes(include=[np.number])
            Xte_num = ds.X_test.select_dtypes(include=[np.number])

            threshold = float(getattr(self.cfg.train, "intermittency_threshold", 0.4))
            adi_threshold = float(getattr(self.cfg.train, "intermittency_adi_threshold", 1.32))
            cv2_threshold = float(getattr(self.cfg.train, "intermittency_cv2_threshold", 0.49))
            croston_alpha = float(getattr(self.cfg.train, "croston_alpha", 0.1))
            target_profile = getattr(ds, "target_series_profile", {})
            target_sparsity_ratio = float(target_profile.get("sparsity_ratio", 0.0))
            target_adi = float(target_profile.get("adi", np.nan))
            target_cv2 = float(target_profile.get("cv2", np.nan))
            if np.isfinite(target_adi) and np.isfinite(target_cv2):
                use_sba = (target_adi >= adi_threshold) and (target_cv2 < cv2_threshold)
            else:
                use_sba = target_sparsity_ratio > threshold
            recency_gamma = float(getattr(self.cfg.cv, "recency_gamma", 0.08))
            target_mask = np.asarray(
                getattr(ds, "train_is_target", pd.Series([True] * len(ds.y_train))),
                dtype=bool,
            )

            # Base model 1: LGBM (current Forecaster model)
            fore.fit(
                Xtr_num,
                ds.y_train,
                dates=getattr(ds, "train_dates", None),
                recency_gamma=recency_gamma,
            )
            yhat_test_lgbm, _ = fore.predict(ds.X_test, ds.y_test, error_fun=MAPE)
            yhat_train_lgbm, _ = fore.predict(ds.X_train, ds.y_train, error_fun=MAPE)

            # Base model 2: Random Forest (for future stacking)
            rf_params = dict(getattr(self.cfg, "rf_model", {}))
            rf_fore = RandomForestForecaster(params=rf_params)
            rf_fore.fit(
                Xtr_num,
                ds.y_train,
                dates=getattr(ds, "train_dates", None),
                recency_gamma=recency_gamma,
            )
            yhat_test_rf, _ = rf_fore.predict(ds.X_test, ds.y_test, error_fun=MAPE)
            yhat_train_rf, _ = rf_fore.predict(ds.X_train, ds.y_train, error_fun=MAPE)

            if use_sba:
                y_train_target = ds.y_train[target_mask].reset_index(drop=True)
                sba = CrostonSBAForecaster(alpha=croston_alpha)
                sba.fit(y_train_target.to_numpy())
                yhat_test = sba.predict(len(ds.y_test))
                yhat_train = sba.predict_in_sample()
                route_model = "croston_sba"
            else:
                yhat_test = yhat_test_lgbm
                yhat_train = yhat_train_lgbm
                route_model = str(self.cfg.model.name)

            # Inverse-transform from scaled space back to original demand units.
            # series_scale is set in dataset.process_data(); default 1.0 if scaling disabled.
            _scale = getattr(ds, "series_scale", 1.0)
            yhat_test  = np.asarray(yhat_test)  * _scale
            yhat_train = np.asarray(yhat_train) * _scale
            yhat_test_lgbm = np.asarray(yhat_test_lgbm) * _scale
            yhat_train_lgbm = np.asarray(yhat_train_lgbm) * _scale
            yhat_test_rf = np.asarray(yhat_test_rf) * _scale
            yhat_train_rf = np.asarray(yhat_train_rf) * _scale

            meta_coef_lgbm = np.nan
            meta_coef_rf = np.nan
            if ensemble_enabled and (not use_sba) and len(meta_history_rows) >= 8:
                meta = EnsembleForecaster(alpha=ridge_alpha)
                hist = pd.DataFrame(meta_history_rows)
                meta.fit(
                    hist["y_pred_lgbm"].to_numpy(),
                    hist["y_pred_rf"].to_numpy(),
                    hist["y_true"].to_numpy(),
                )
                if meta.is_fitted:
                    yhat_test = np.maximum(0.0, meta.predict(yhat_test_lgbm, yhat_test_rf))
                    yhat_train = np.maximum(0.0, meta.predict(yhat_train_lgbm, yhat_train_rf))
                    route_model = "ridge_stack_lgbm_rf"
                    meta_coef_lgbm = float(meta.coef_[0])
                    meta_coef_rf = float(meta.coef_[1])

            y_true = np.asarray(ds.y_test) * _scale

            # Conformal-style residual calibration:
            # - for early cuts, bootstrap from target train residuals
            # - once available, use accumulated out-of-fold residual history
            if use_sba:
                y_train_for_stock = y_train_target
                yhat_train_for_stock = np.asarray(yhat_train)
            else:
                y_train_for_stock = ds.y_train[target_mask].reset_index(drop=True)
                yhat_train_for_stock = np.asarray(yhat_train)[target_mask]

            if len(residual_history) >= 4:
                residual_q = self._residual_quantiles(np.asarray(residual_history))
            else:
                seed = np.asarray(y_train_for_stock, dtype=float) * _scale
                naive_residuals = seed[1:] - seed[:-1] if seed.size >= 2 else np.zeros(1, dtype=float)
                residual_q = self._residual_quantiles(naive_residuals)

            yhat_p10 = np.maximum(0.0, yhat_test + residual_q["q10"])
            yhat_p50 = np.maximum(0.0, yhat_test + residual_q["q50"])
            yhat_p90 = np.maximum(0.0, yhat_test + residual_q["q90"])

            # 4) metriche "classiche"
            wape_num = float(np.sum(np.abs(y_true - yhat_test)))
            wape_den = float(np.sum(np.abs(y_true)))
            metrics = {
                "MAPE": float(MAPE(y_true, yhat_test)),
                "WAPE": float(WAPE(y_true, yhat_test)),
                "SMAPE": float(SMAPE(y_true, yhat_test)),
                "MASE": float(MASE(y_true, yhat_test, y_insample=np.asarray(y_train_for_stock) * _scale)),
                "WAPE_num": wape_num,
                "WAPE_den": wape_den,
                "Q10_resid": float(residual_q["q10"]),
                "Q50_resid": float(residual_q["q50"]),
                "Q90_resid": float(residual_q["q90"]),
                "MetaCoefLGBM": float(meta_coef_lgbm) if np.isfinite(meta_coef_lgbm) else np.nan,
                "MetaCoefRF": float(meta_coef_rf) if np.isfinite(meta_coef_rf) else np.nan,
            }

            # 5) Stock analysis per cut: cartella dedicata
            cut_dir = os.path.join(self.output_dir, f"{item}", f"cut_{cut_idx}")
            os.makedirs(cut_dir, exist_ok=True)

            stock_df = fore.stock_analysis(
                y_test=ds.y_test * _scale,
                y_pred_test=pd.Series(yhat_test, index=ds.y_test.index),
                y_train=y_train_for_stock * _scale,
                y_pred_train=pd.Series(yhat_train_for_stock),
                sell_in_true=ds.X_test.get("orders_SellIn", ds.y_test) if hasattr(ds, "X_test") else None,
                safety_stock_time=getattr(self.cfg.stock, "safety_stock_window", 4),
                lead_time=getattr(self.cfg.stock, "lead_time", 1),
                expected_service_level=getattr(self.cfg.stock, "service_level", 0.95),
                product_name=item,
                save_fp=cut_dir,
                train_on_log=False,
                residual_quantiles=residual_q,
                apply_bias_correction=not use_sba,
            )

            stock_df["y_pred_p10"] = yhat_p10
            stock_df["y_pred_p50"] = yhat_p50
            stock_df["y_pred_p90"] = yhat_p90

            # 6) log extra metriche dalla stock analysis
            try:
                metrics["ServiceLevel"] = float(stock_df["service_level"].iloc[0])
            except Exception:
                pass
            try:
                metrics["SafetyStock"] = float(stock_df["safety_stock"].iloc[0])
            except Exception:
                pass

            # 7) Preds "timelined" per report
            test_dates = self._get_test_dates(ds)
            for t, yp, yt, p10, p50, p90, yp_lgbm, yp_rf in zip(
                test_dates, yhat_test, y_true, yhat_p10, yhat_p50, yhat_p90, yhat_test_lgbm, yhat_test_rf
            ):
                artifacts_rows.append({
                    "item": item,
                    "cut_idx": cut_idx,
                    "timestamp": str(pd.to_datetime(t).date()) if not pd.isna(t) else "",
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "y_pred_lgbm": float(yp_lgbm),
                    "y_pred_rf": float(yp_rf),
                    "y_pred_p10": float(p10),
                    "y_pred_p50": float(p50),
                    "y_pred_p90": float(p90),
                })
                oof_rows.append({
                    "item": item,
                    "cut_idx": cut_idx,
                    "timestamp": str(pd.to_datetime(t).date()) if not pd.isna(t) else "",
                    "y_true": float(yt),
                    "y_pred_lgbm": float(yp_lgbm),
                    "y_pred_rf": float(yp_rf),
                })
                meta_history_rows.append({
                    "y_true": float(yt),
                    "y_pred_lgbm": float(yp_lgbm),
                    "y_pred_rf": float(yp_rf),
                })
            residual_history.extend((y_true - yhat_test).tolist())

            # 8) Artefatti salvati (CSV + XLSX della stock analysis) & MLflow nested run
            preds_csv_fp = os.path.join(cut_dir, f"preds_{item}_cut{cut_idx}.csv")
            pd.DataFrame(artifacts_rows).to_csv(preds_csv_fp, index=False)

            # XLSX della stock analysis per cut
            stock_xlsx_fp = os.path.join(cut_dir, f"stock_analysis_{item}_cut{cut_idx}.xlsx")
            try:
                stock_df.to_excel(stock_xlsx_fp, index=False)
            except Exception:
                # fallback in caso di engine assente: salva CSV
                stock_xlsx_fp = os.path.join(cut_dir, f"stock_analysis_{item}_cut{cut_idx}.csv")
                stock_df.to_csv(stock_xlsx_fp, index=False)

            with mlflow.start_run(run_name=f"{item}_cut_{cut_idx}", nested=True):
                mlflow.log_params({
                    "item": item,
                    "cut_idx": cut_idx,
                    "boundary": str(cut.boundary.date()),
                    "test_len": str(cut.test_len),
                    "model": self.cfg.model.name,
                    "route_model": route_model,
                    "target_sparsity_ratio": target_sparsity_ratio,
                    "target_adi": target_adi,
                    "target_cv2": target_cv2,
                    "intermittency_threshold": threshold,
                    "intermittency_adi_threshold": adi_threshold,
                    "intermittency_cv2_threshold": cv2_threshold,
                    "ensemble_enabled": ensemble_enabled,
                    "ridge_alpha": ridge_alpha,
                })
                mlflow.log_metrics(metrics)

                # Immagini create da stock_analysis:
                png_test = os.path.join(cut_dir, f"{item}_stock_analysis.png")
                png_train = os.path.join(cut_dir, f"{item}_stock_analysis_train.png")

                if os.path.exists(png_test):
                    mlflow.log_artifact(png_test, artifact_path=f"{item}/cut_{cut_idx}")
                if os.path.exists(png_train):
                    mlflow.log_artifact(png_train, artifact_path=f"{item}/cut_{cut_idx}")

                # Preds CSV e Stock XLSX
                mlflow.log_artifact(preds_csv_fp, artifact_path=f"{item}/cut_{cut_idx}")
                mlflow.log_artifact(stock_xlsx_fp, artifact_path=f"{item}/cut_{cut_idx}")

            results.append(BacktestResult(
                item=item,
                cut_idx=cut_idx,
                boundary=str(cut.boundary.date()),
                metrics=metrics,
                y_true=y_true.tolist(),
                y_pred=[float(v) for v in yhat_test],
                cut_dir=cut_dir,
            ))

        # 9) Aggregazione per item
        agg = pd.DataFrame([r.metrics for r in results])
        wape_num_sum = float(np.nansum(agg["WAPE_num"]))
        wape_den_sum = float(np.nansum(agg["WAPE_den"]))
        wape_pooled = float(100.0 * wape_num_sum / wape_den_sum) if wape_den_sum > 1e-12 else float("nan")

        summary = {
            "item": item,
            "cuts": len(results),
            "MAPE_mean": float(agg["MAPE"].mean()), "MAPE_std": float(agg["MAPE"].std(ddof=1)),
            "WAPE_mean": wape_pooled, "WAPE_std": float(agg["WAPE"].std(ddof=1)),
            "WAPE_num": wape_num_sum, "WAPE_den": wape_den_sum,
            "SMAPE_mean": float(agg["SMAPE"].mean()), "SMAPE_std": float(agg["SMAPE"].std(ddof=1)),
            "MASE_mean": float(agg["MASE"].mean()), "MASE_std": float(agg["MASE"].std(ddof=1)),
        }

        # File “preds” consolidato per item (ultimo stato di artifacts_rows)
        item_preds_csv = os.path.join(self.output_dir, f"preds_{item}.csv")
        pd.DataFrame(artifacts_rows).to_csv(item_preds_csv, index=False)
        item_oof_csv = os.path.join(self.output_dir, f"oof_base_models_{item}.csv")
        pd.DataFrame(oof_rows).to_csv(item_oof_csv, index=False)

        return {
            "item": item,
            "cuts": [r.__dict__ for r in results],
            "summary": summary,
            "preds_csv": item_preds_csv,
            "oof_csv": item_oof_csv,
        }

    def run_all(self, items: List[str]) -> Dict[str, Any]:
        run_payload = {"items": [], "global_summary": {}}
        all_summaries = []

        with mlflow.start_run(run_name="backtest_multi_item"):
            mlflow.log_params({
                "model": self.cfg.model.name,
                "granularity": self.cfg.data.granularity,
                "nr_horizon": self.cfg.data.nr_horizon,
                "train_len": self.cfg.train.train_len,
                "test_len": self.cfg.train.test_len,
                "n_cuts": self.n_cuts,
            })
            # salva l'Hydra config come artifact
            try:
                mlflow.log_dict(json.loads(self.cfg.to_json()), "hydra_config.json")
            except Exception:
                pass

            for it in items:
                res = self.run_for_item(it)
                run_payload["items"].append(res)
                all_summaries.append(res["summary"])
                mlflow.log_metrics({f"{it}_MAPE_mean": res["summary"]["MAPE_mean"]})

            # global summary
            gdf = pd.DataFrame(all_summaries)
            global_wape_num = float(np.nansum(gdf["WAPE_num"]))
            global_wape_den = float(np.nansum(gdf["WAPE_den"]))
            global_wape_mean = float(100.0 * global_wape_num / global_wape_den) if global_wape_den > 1e-12 else float("nan")
            global_summary = {
                "items": len(items),
                "MAPE_mean": float(gdf["MAPE_mean"].mean()),
                "MAPE_std": float(gdf["MAPE_mean"].std(ddof=1)),
                "WAPE_mean": global_wape_mean,
                "WAPE_std": float(gdf["WAPE_mean"].std(ddof=1)),
                "WAPE_num": global_wape_num,
                "WAPE_den": global_wape_den,
                "SMAPE_mean": float(gdf["SMAPE_mean"].mean()),
                "SMAPE_std": float(gdf["SMAPE_mean"].std(ddof=1)),
                "MASE_mean": float(gdf["MASE_mean"].mean()),
                "MASE_std": float(gdf["MASE_mean"].std(ddof=1)),
            }
            run_payload["global_summary"] = global_summary
            mlflow.log_metrics({f"GLOBAL_{k}": v for k, v in global_summary.items()})

        return run_payload

