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

import hydra
from omegaconf import DictConfig
from data.dataset import SalesDataset
from model.forecaster import Forecaster
from backtest import BacktestRunner
import numpy as np
from utils.helper import resolve_target_keys


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
        runner.run_all(items)
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
