# backend/main.py
# backend/main.py
import hydra
from omegaconf import DictConfig
from data.dataset import SalesDataset
from model.forecaster import Forecaster
from backtest import BacktestRunner
import numpy as np
from utils.helper import resolve_target_keys
#from reporting import build_html_report

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # 0) Prepara i blocchi di config con le interpolazioni già risolte
    prepare_config = resolve_target_keys(cfg)

    # 1) Dataset
    dataset_cfg = getattr(cfg, "data", {})
    dataset_extra_cfg = getattr(cfg, "dataset_config", {})  # opzionale nel tuo YAML

    if cfg.backtest.enabled:
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
        #report_fp = build_html_report(payload, out_dir=cfg.backtest.output_dir, report_name="report.html")
        #print(f"✅ Backtest finito. Report: {report_fp}")
        return

    # run singolo (come fallback)
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
    dataset.process_data(product=cfg.data.product_name,
                         train_params={"train_len": cfg.train.train_len,
                                       "test_len": cfg.train.test_len,
                                       "boundary": cfg.train.boundary})

    forecaster = Forecaster(model_name=cfg.model.name, params=cfg.model.params)
    forecaster.model.fit(dataset.X_train.select_dtypes(include=[np.number]), dataset.y_train)
    _ = forecaster.predict(dataset.X_test, dataset.y_test)
    print("✅ Run singolo completato.")

if __name__ == "__main__":
    main()




import os
import hydra
from omegaconf import DictConfig

from data.dataset import SalesDataset
from model.forecaster import Forecaster
from utils.metrics import MAPE
from utils.helper import resolve_target_keys
from utils.logging_utils import setup_logger

logger = setup_logger("main")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 0) Prepara i blocchi di config con le interpolazioni già risolte
    prepare_config = resolve_target_keys(cfg)

    # 1) Dataset
    dataset_cfg = getattr(cfg, "data", {})
    dataset_extra_cfg = getattr(cfg, "dataset_config", {})  # opzionale nel tuo YAML
    # Esempi utili in dataset_extra_cfg:
    #   debug_save: false
    #   known_future_promos: false
    #   market_lags: [1,4,8,12]
    #   market_trend_windows: [4,8]
    #   per_series_scaling: { enabled: true, method: zscore, windows: [8,26] }

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
        product=dataset_cfg.product_name,
        train_params={
            "train_len": cfg.train.train_len,
            "test_len": cfg.train.test_len,
            "boundary": cfg.train.boundary,
        },
    )

    # 2) Forecaster + Grid Search con CV rolling sulla target (Soluzione B)
    fore = Forecaster(model_name=cfg.model.name, params=cfg.model.params)
    rec_gamma = getattr(cfg, "cv", {}).get("recency_gamma", 0.08)  # o hardcode 0.08
    fore.fit(dataset.X_train, dataset.y_train, dates=dataset.train_dates, recency_gamma=rec_gamma)

    # Scegli la colonna prodotto da usare per i folds: se c'è product_id, preferiscilo
    product_var_for_cv = "product_id" if "product_id" in dataset.X_train.columns else dataset_cfg.product_column


    # 3) Predict (test holdout)
    y_pred_test, mape = fore.predict(dataset.X_test, dataset.y_test, error_fun=MAPE, train_on_log=getattr(cfg.model, "train_on_log", False))
    print(f"MAPE TEST: {mape:.2f}%")

    # 4) Predizioni su train (stesso spazio delle predizioni test)
    y_pred_train, _ = fore.predict(dataset.X_train, dataset.y_train, error_fun=MAPE, train_on_log=getattr(cfg.model, "train_on_log", False))

    # 5) Stock analysis / salvataggi
    stock_cfg = getattr(cfg, "stock", {})
    output_dir = getattr(stock_cfg, "output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df_stock = fore.stock_analysis(
        y_test=dataset.y_test,
        y_pred_test=y_pred_test,
        y_train=dataset.y_train,
        y_pred_train=y_pred_train,
        sell_in_true=dataset.X_test.get("orders_SellIn", dataset.y_test),
        safety_stock_time=stock_cfg.safety_stock_window,
        lead_time=stock_cfg.lead_time,
        expected_service_level=stock_cfg.service_level,
        product_name=dataset_cfg.product_name,
        save_fp=output_dir,
        train_on_log=getattr(cfg.model, "train_on_log", False),
    )

    df_path = os.path.join(output_dir, f"{dataset_cfg.product_name}_stock_table.xlsx")
    df_stock.to_excel(df_path, index=False)
    logger.info(f"Saved stock analysis table → {df_path}")

if __name__ == "__main__":
    main()
