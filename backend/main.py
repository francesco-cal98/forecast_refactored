# =============================================================================
# main.py -- Entry point for the forecasting pipeline
#
# HOW IT WORKS:
#   Hydra reads config/config.yaml and injects it as `cfg`. Two modes:
#
#   1. BACKTEST mode (cfg.backtest.enabled = true)  <- default
#      Loops over every product, runs BacktestRunner which trains on N
#      historical "cuts" and evaluates each one. Saves predictions, stock
#      analysis charts, and logs metrics to MLflow.
#
#   2. SINGLE-RUN mode (cfg.backtest.enabled = false)
#      Trains on one product, predicts the test window, prints WAPE.
#      Useful for quick checks and debugging.
#
# HOW TO RUN:
#   cd backend
#   python main.py                              # backtest, xlsx_sample, ALPHA
#   python main.py data.product_name=BETA       # different product
#   python main.py data=csv_history             # switch to CSV data source
#   python main.py backtest.enabled=false       # single-run mode
#   python main.py model=light_lgm              # switch model to LightGBM
# =============================================================================

import datetime
import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig
from data.dataset import SalesDataset
from model.forecaster import Forecaster
from backtest import BacktestRunner
import numpy as np
from utils.helper import resolve_target_keys


def _save_benchmark_snapshot(payload: dict, cfg: DictConfig) -> None:
    """Append one line to results/benchmark_log.jsonl after every backtest run.
    This gives a permanent, timestamped record of model performance so we can
    measure the delta introduced by each code change.
    """
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / "benchmark_log.jsonl"

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": git_hash,
        "model": str(cfg.model.name),
        "n_cuts": int(cfg.backtest.n_cuts),
        "train_len": str(cfg.train.train_len),
        "test_len": str(cfg.train.test_len),
        **payload["global_summary"],
        "per_item": {r["item"]: r["summary"] for r in payload["items"]},
    }

    def _clean(obj):
        """Recursively replace NaN/Inf with None for valid JSON serialization."""
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(log_path, "a") as f:
        f.write(json.dumps(_clean(snapshot)) + "\n")

    gs = payload["global_summary"]
    print(
        f"\nBenchmark snapshot saved -> {log_path}"
        f"\n  git={git_hash}  model={cfg.model.name}"
        f"  WAPE={gs['WAPE_mean']:.2f}% (+/-{gs['WAPE_std']:.2f})"
        f"  MAPE={gs['MAPE_mean']:.2f}%"
        f"  MASE={gs['MASE_mean']:.2f}"
    )


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Resolve ${data.target_column} placeholders inside the prepare block.
    # Hydra does not interpolate inside nested mapping keys automatically.
    prepare_config = resolve_target_keys(cfg)

    dataset_cfg       = getattr(cfg, "data", {})
    dataset_extra_cfg = getattr(cfg, "dataset_config", {})

    # ------------------------------------------------------------------
    # MODE 1: BACKTEST (default, backtest.enabled = true)
    # ------------------------------------------------------------------
    if cfg.backtest.enabled:
        # Load full dataset once just to get list of all products
        ds = SalesDataset(
            csv_file=dataset_cfg.file_path,
            separator=dataset_cfg.separator,
            date_column=dataset_cfg.date_column,
            target_column=dataset_cfg.target_column,
            nr_horizon=dataset_cfg.nr_horizon,
            prepare_config=prepare_config,
            column_selection=cfg.column_selection,
            product_var=dataset_cfg.product_column,
            product_name=dataset_cfg.product_name,
            granularity=dataset_cfg.granularity,
            config=dataset_extra_cfg,
        )
        items = list(ds.get_all_products())

        runner = BacktestRunner(
            dataset_cls=SalesDataset,
            forecaster_cls=Forecaster,
            cfg=cfg,
            tracking_uri=cfg.mlflow.tracking_uri,
            experiment_name=cfg.mlflow.experiment_name,
            output_dir=cfg.backtest.output_dir,
            n_cuts=cfg.backtest.n_cuts,
        )
        payload = runner.run_all(items)
        _save_benchmark_snapshot(payload, cfg)
        return

    # ------------------------------------------------------------------
    # MODE 2: SINGLE RUN (backtest.enabled = false)
    # ------------------------------------------------------------------
    dataset = SalesDataset(
        csv_file=dataset_cfg.file_path,
        separator=dataset_cfg.separator,
        date_column=dataset_cfg.date_column,
        target_column=dataset_cfg.target_column,
        nr_horizon=dataset_cfg.nr_horizon,
        prepare_config=prepare_config,
        column_selection=cfg.column_selection,
        product_var=dataset_cfg.product_column,
        product_name=dataset_cfg.product_name,
        granularity=dataset_cfg.granularity,
        config=dataset_extra_cfg,
    )

    dataset.process_data(
        product=cfg.data.product_name,
        train_params={
            "train_len": cfg.train.train_len,
            "test_len":  cfg.train.test_len,
            "boundary":  cfg.train.boundary,
        }
    )

    # Always call Forecaster.fit() -- not model.fit() directly -- so that
    # temporal exponential-decay sample weights are computed and passed through
    forecaster    = Forecaster(model_name=cfg.model.name, params=cfg.model.params)
    recency_gamma = float(getattr(cfg.cv, "recency_gamma", 0.08))
    forecaster.fit(
        dataset.X_train,
        dataset.y_train,
        dates=getattr(dataset, "train_dates", None),
        recency_gamma=recency_gamma,
    )

    # predict() returns predictions in scaled space (targets were divided by
    # series_scale inside process_data). Multiply back to get original units.
    y_pred, err = forecaster.predict(dataset.X_test, dataset.y_test)
    _scale = getattr(dataset, "series_scale", 1.0)
    y_pred = y_pred * _scale

    print(f"Single run complete. WAPE: {err:.2f}%")


if __name__ == "__main__":
    main()
