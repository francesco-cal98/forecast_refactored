# backend/backtest.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from utils.helper import resolve_config_refs,resolve_target_keys
from utils.metrics import MAPE,WAPE,MASE,SMAPE  # già nel tuo repo
from utils.plots import plot_coverage, plot_error_vs_horizon


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

    @staticmethod
    def _compute_pooled_wape(abs_error_sum: float, abs_actual_sum: float) -> Tuple[float, bool]:
        """
        Compute pooled WAPE across all evaluation points:
        100 * sum(|y - yhat|) / sum(|y|).
        Returns (value, valid_flag). If denominator is 0, returns (NaN, False).
        """
        if abs_actual_sum == 0:
            return float("nan"), False
        return float(100.0 * abs_error_sum / abs_actual_sum), True

    @staticmethod
    def _aggregate_probabilistic_diagnostics(
        y_true_cuts: List[np.ndarray],
        p50_cuts: List[np.ndarray],
        p10_cuts: List[Optional[np.ndarray]],
        p90_cuts: List[Optional[np.ndarray]],
    ) -> Dict[str, Any]:
        abs_error_by_h_values: List[List[float]] = []
        coverage_by_h_values: List[List[float]] = []
        coverage_hits: List[float] = []

        for y_true, p50, p10, p90 in zip(y_true_cuts, p50_cuts, p10_cuts, p90_cuts):
            y_true = np.asarray(y_true)
            p50 = np.asarray(p50)

            abs_errors = np.abs(y_true - p50)
            for h_idx, err in enumerate(abs_errors):
                if h_idx >= len(abs_error_by_h_values):
                    abs_error_by_h_values.append([])
                abs_error_by_h_values[h_idx].append(float(err))

            if p10 is not None and p90 is not None:
                lower = np.minimum(np.asarray(p10), np.asarray(p90))
                upper = np.maximum(np.asarray(p10), np.asarray(p90))
                in_band = ((y_true >= lower) & (y_true <= upper)).astype(float)

                for h_idx, hit in enumerate(in_band):
                    if h_idx >= len(coverage_by_h_values):
                        coverage_by_h_values.append([])
                    coverage_by_h_values[h_idx].append(float(hit))
                coverage_hits.extend(in_band.tolist())

        abs_error_by_h = [
            float(np.mean(vals)) if len(vals) > 0 else float("nan")
            for vals in abs_error_by_h_values
        ]
        coverage_by_h = (
            [float(np.mean(vals)) if len(vals) > 0 else float("nan") for vals in coverage_by_h_values]
            if len(coverage_by_h_values) > 0
            else None
        )
        coverage_available = len(coverage_hits) > 0
        coverage_10_90 = float(np.mean(coverage_hits)) if coverage_available else float("nan")

        return {
            "abs_error_by_h": abs_error_by_h,
            "coverage_by_h": coverage_by_h,
            "coverage_10_90": coverage_10_90,
            "coverage_available": coverage_available,
        }

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
        abs_error_sum = 0.0
        abs_actual_sum = 0.0
        y_true_cuts: List[np.ndarray] = []
        p50_cuts: List[np.ndarray] = []
        p10_cuts: List[Optional[np.ndarray]] = []
        p90_cuts: List[Optional[np.ndarray]] = []

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

            # Use Forecaster.fit() so temporal sample weights are applied correctly
            recency_gamma = float(getattr(self.cfg.cv, "recency_gamma", 0.08))
            fore.fit(
                Xtr_num,
                ds.y_train,
                dates=getattr(ds, "train_dates", None),
                recency_gamma=recency_gamma,
            )

            # pred test
            yhat_test, _ = fore.predict(ds.X_test, ds.y_test, error_fun=MAPE)
            # pred train (per stock_analysis)
            yhat_train = fore.model.predict(Xtr_num)

            # Inverse-transform from scaled space back to original demand units.
            # series_scale is set in dataset.process_data(); default 1.0 if scaling disabled.
            _scale = getattr(ds, "series_scale", 1.0)
            yhat_test  = yhat_test  * _scale
            yhat_train = yhat_train * _scale

            y_true = np.asarray(ds.y_test) * _scale
            p50 = np.asarray(yhat_test)
            p10 = None
            p90 = None
            try:
                residuals_train = (np.asarray(ds.y_train) * _scale) - np.asarray(yhat_train)
                if residuals_train.size > 0:
                    q10, q90 = np.nanquantile(residuals_train, [0.1, 0.9])
                    p10 = p50 + q10
                    p90 = p50 + q90
            except Exception:
                pass

            y_true_cuts.append(y_true)
            p50_cuts.append(p50)
            p10_cuts.append(p10)
            p90_cuts.append(p90)

            # 4) metriche "classiche"
            metrics = {
                "MAPE": float(MAPE(y_true, yhat_test)),
                "WAPE": float(WAPE(y_true, yhat_test)),
                "SMAPE": float(SMAPE(y_true, yhat_test)),
                "MASE": float(MASE(y_true, yhat_test)),
            }
            abs_error_sum += float(np.sum(np.abs(y_true - yhat_test)))
            abs_actual_sum += float(np.sum(np.abs(y_true)))

            # 5) Stock analysis per cut: cartella dedicata
            cut_dir = os.path.join(self.output_dir, f"{item}", f"cut_{cut_idx}")
            os.makedirs(cut_dir, exist_ok=True)

            stock_df = fore.stock_analysis(
                y_test=ds.y_test * _scale,
                y_pred_test=pd.Series(yhat_test, index=ds.y_test.index),
                y_train=ds.y_train * _scale,
                y_pred_train=pd.Series(yhat_train, index=ds.y_train.index),
                sell_in_true=ds.X_test.get("orders_SellIn", ds.y_test) if hasattr(ds, "X_test") else None,
                safety_stock_time=getattr(self.cfg.stock, "safety_stock_window", 4),
                lead_time=getattr(self.cfg.stock, "lead_time", 1),
                expected_service_level=getattr(self.cfg.stock, "service_level", 0.95),
                product_name=item,
                save_fp=cut_dir,
                train_on_log=False,
            )

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
            for h_idx, (t, yp, yt) in enumerate(zip(test_dates, yhat_test, y_true), start=1):
                artifacts_rows.append({
                    "item": item,
                    "cut_idx": cut_idx,
                    "h": h_idx,
                    "timestamp": str(pd.to_datetime(t).date()) if not pd.isna(t) else "",
                    "y_true": float(yt),
                    "y_pred": float(yp),  # backward-compatible alias for p50
                    "p50": float(yp),
                    "p10": float(p10[h_idx - 1]) if p10 is not None else float("nan"),
                    "p90": float(p90[h_idx - 1]) if p90 is not None else float("nan"),
                })

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
                })
                mlflow.log_metrics(metrics)

                # Immagini create da stock_analysis:
                png_test = os.path.join(cut_dir, f"{item}_stock_analysis.png")
                png_train = os.path.join(cut_dir, f"{item}_stock_analysis_train.png")
                png_inventory_band = os.path.join(cut_dir, f"{item}_inventory_scenario_band.png")

                if os.path.exists(png_test):
                    mlflow.log_artifact(png_test, artifact_path=f"{item}/cut_{cut_idx}")
                if os.path.exists(png_train):
                    mlflow.log_artifact(png_train, artifact_path=f"{item}/cut_{cut_idx}")
                if os.path.exists(png_inventory_band):
                    mlflow.log_artifact(png_inventory_band, artifact_path=f"{item}/cut_{cut_idx}")

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
        wape_pooled, wape_pooled_valid = self._compute_pooled_wape(abs_error_sum, abs_actual_sum)
        prob_diag = self._aggregate_probabilistic_diagnostics(
            y_true_cuts=y_true_cuts,
            p50_cuts=p50_cuts,
            p10_cuts=p10_cuts,
            p90_cuts=p90_cuts,
        )
        abs_error_by_h = prob_diag["abs_error_by_h"]
        coverage_by_h = prob_diag["coverage_by_h"]
        coverage_10_90 = prob_diag["coverage_10_90"]
        coverage_available = prob_diag["coverage_available"]

        item_dir = os.path.join(self.output_dir, f"{item}")
        os.makedirs(item_dir, exist_ok=True)
        n_h = len(abs_error_by_h)
        coverage_series = coverage_by_h if coverage_by_h is not None else [float("nan")] * n_h
        prob_diag_csv = os.path.join(item_dir, f"prob_diagnostics_{item}.csv")
        pd.DataFrame({
            "h": np.arange(1, n_h + 1),
            "abs_error_by_h": abs_error_by_h,
            "coverage_10_90_by_h": coverage_series,
        }).to_csv(prob_diag_csv, index=False)

        fig_err = plot_error_vs_horizon(item, abs_error_by_h)
        error_plot_fp = os.path.join(item_dir, f"{item}_error_vs_horizon.png")
        fig_err.savefig(error_plot_fp, bbox_inches="tight")
        plt.close(fig_err)

        coverage_plot_fp = None
        if coverage_available:
            fig_cov = plot_coverage(
                product=item,
                coverage_global=coverage_10_90,
                expected=0.80,
                coverage_by_h=coverage_by_h,
            )
            coverage_plot_fp = os.path.join(item_dir, f"{item}_coverage_10_90.png")
            fig_cov.savefig(coverage_plot_fp, bbox_inches="tight")
            plt.close(fig_cov)

        summary = {
            "item": item,
            "cuts": len(results),
            "MAPE_mean": float(agg["MAPE"].mean()), "MAPE_std": float(agg["MAPE"].std(ddof=1)),
            "WAPE_mean": wape_pooled, "WAPE_std": float(agg["WAPE"].std(ddof=1)),
            "WAPE_pooled": wape_pooled, "WAPE_pooled_valid": wape_pooled_valid,
            "WAPE_mean_of_cuts": float(agg["WAPE"].mean()),
            "coverage_10_90": coverage_10_90,
            "SMAPE_mean": float(agg["SMAPE"].mean()), "SMAPE_std": float(agg["SMAPE"].std(ddof=1)),
            "MASE_mean": float(agg["MASE"].mean()), "MASE_std": float(agg["MASE"].std(ddof=1)),
        }

        # File “preds” consolidato per item (ultimo stato di artifacts_rows)
        item_preds_csv = os.path.join(self.output_dir, f"preds_{item}.csv")
        pd.DataFrame(artifacts_rows).to_csv(item_preds_csv, index=False)

        return {
            "item": item,
            "cuts": [r.__dict__ for r in results],
            "summary": summary,
            "preds_csv": item_preds_csv,
            "wape_abs_error_sum": abs_error_sum,
            "wape_abs_actual_sum": abs_actual_sum,
            "prob_diag_csv": prob_diag_csv,
            "coverage_plot": coverage_plot_fp,
            "error_vs_horizon_plot": error_plot_fp,
        }

    def run_all(self, items: List[str]) -> Dict[str, Any]:
        run_payload = {"items": [], "global_summary": {}}
        all_summaries = []
        global_abs_error_sum = 0.0
        global_abs_actual_sum = 0.0

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
                global_abs_error_sum += float(res.get("wape_abs_error_sum", 0.0))
                global_abs_actual_sum += float(res.get("wape_abs_actual_sum", 0.0))
                mlflow.log_metrics({f"{it}_MAPE_mean": res["summary"]["MAPE_mean"]})
                for fp in [res.get("prob_diag_csv"), res.get("coverage_plot"), res.get("error_vs_horizon_plot")]:
                    if fp and os.path.exists(fp):
                        mlflow.log_artifact(fp, artifact_path=f"{it}/diagnostics")

            # global summary
            gdf = pd.DataFrame(all_summaries)
            global_wape_pooled, global_wape_pooled_valid = self._compute_pooled_wape(global_abs_error_sum, global_abs_actual_sum)
            global_summary = {
                "items": len(items),
                "MAPE_mean": float(gdf["MAPE_mean"].mean()),
                "MAPE_std": float(gdf["MAPE_mean"].std(ddof=1)),
                "WAPE_mean": global_wape_pooled,
                "WAPE_std": float(gdf["WAPE_mean"].std(ddof=1)),
                "WAPE_pooled": global_wape_pooled,
                "WAPE_pooled_valid": global_wape_pooled_valid,
                "WAPE_mean_of_items": float(gdf["WAPE_mean"].mean()),
                "SMAPE_mean": float(gdf["SMAPE_mean"].mean()),
                "SMAPE_std": float(gdf["SMAPE_mean"].std(ddof=1)),
                "MASE_mean": float(gdf["MASE_mean"].mean()),
                "MASE_std": float(gdf["MASE_mean"].std(ddof=1)),
            }
            run_payload["global_summary"] = global_summary
            mlflow.log_metrics({f"GLOBAL_{k}": v for k, v in global_summary.items()})

        return run_payload
