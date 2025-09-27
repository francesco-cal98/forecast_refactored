# backend/backtest.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import mlflow

from utils.metrics import MAPE,WAPE,MASE,SMAPE  # già nel tuo repo


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

    # ---------- main API ----------
    def run_for_item(self, item: str) -> Dict[str, Any]:
        # 1) Sonda le date disponibili per costruire i cut
        ds_probe = self.dataset_cls(
            csv_file=self.cfg.data.file_path,
            separator=self.cfg.data.separator,
            date_column=self.cfg.data.date_column,
            target_column=self.cfg.data.target_column,
            nr_horizon=self.cfg.data.nr_horizon,
            prepare_config=self.cfg.prepare,
            column_selection=self.cfg.column_selection,
            product_var=self.cfg.data.product_column,
            granularity=self.cfg.data.granularity,
            product_name=item,
            config=getattr(self.cfg, "dataset_config", {}) or {},
        )
        df_full = ds_probe._prepare_dataset().sort_values([ds_probe.product_var, ds_probe.date_column])
        df_full = ds_probe._encode_time(df_full, time_unit=ds_probe.granularity)
        tgt_dates = df_full.loc[df_full[ds_probe.product_var] == item, ds_probe.date_column]
        cuts = self._build_cuts(tgt_dates, self.cfg.train.test_len, self.n_cuts)

        # Artefatti complessivi per questo item
        results: List[BacktestResult] = []
        artifacts_rows = []

        for cut_idx, cut in enumerate(cuts, start=1):
            # 2) Prepara il dataset per questo cut
            ds = self.dataset_cls(
                csv_file=self.cfg.data.file_path,
                separator=self.cfg.data.separator,
                date_column=self.cfg.data.date_column,
                target_column=self.cfg.data.target_column,
                nr_horizon=self.cfg.data.nr_horizon,
                prepare_config=self.cfg.prepare,
                column_selection=self.cfg.column_selection,
                product_var=self.cfg.data.product_column,
                granularity=self.cfg.data.granularity,
                product_name=item,
                config=getattr(self.cfg, "dataset_config", {}) or {},
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

            fore.model.fit(Xtr_num, ds.y_train)

            # pred test
            yhat_test, _ = fore.predict(ds.X_test, ds.y_test, error_fun=MAPE)
            # pred train (per stock_analysis)
            yhat_train = fore.model.predict(Xtr_num)

            y_true = np.asarray(ds.y_test)

            # 4) metriche "classiche"
            metrics = {
                "MAPE": float(MAPE(y_true, yhat_test)),
                "WAPE": float(WAPE(y_true, yhat_test)),
                "SMAPE": float(SMAPE(y_true, yhat_test)),
                "MASE": float(MASE(y_true, yhat_test)),
            }

            # 5) Stock analysis per cut: cartella dedicata
            cut_dir = os.path.join(self.output_dir, f"{item}", f"cut_{cut_idx}")
            os.makedirs(cut_dir, exist_ok=True)

            stock_df = fore.stock_analysis(
                y_test=ds.y_test,
                y_pred_test=pd.Series(yhat_test, index=ds.y_test.index),
                y_train=ds.y_train,
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
            for t, yp, yt in zip(test_dates, yhat_test, y_true):
                artifacts_rows.append({
                    "item": item,
                    "cut_idx": cut_idx,
                    "timestamp": str(pd.to_datetime(t).date()) if not pd.isna(t) else "",
                    "y_true": float(yt),
                    "y_pred": float(yp),
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
        summary = {
            "item": item,
            "cuts": len(results),
            "MAPE_mean": float(agg["MAPE"].mean()), "MAPE_std": float(agg["MAPE"].std(ddof=1)),
            "WAPE_mean": float(agg["WAPE"].mean()), "WAPE_std": float(agg["WAPE"].std(ddof=1)),
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
            global_summary = {
                "items": len(items),
                "MAPE_mean": float(gdf["MAPE_mean"].mean()),
                "MAPE_std": float(gdf["MAPE_mean"].std(ddof=1)),
                "WAPE_mean": float(gdf["WAPE_mean"].mean()),
                "WAPE_std": float(gdf["WAPE_mean"].std(ddof=1)),
                "SMAPE_mean": float(gdf["SMAPE_mean"].mean()),
                "SMAPE_std": float(gdf["SMAPE_mean"].std(ddof=1)),
                "MASE_mean": float(gdf["MASE_mean"].mean()),
                "MASE_std": float(gdf["MASE_mean"].std(ddof=1)),
            }
            run_payload["global_summary"] = global_summary
            mlflow.log_metrics({f"GLOBAL_{k}": v for k, v in global_summary.items()})

        return run_payload
