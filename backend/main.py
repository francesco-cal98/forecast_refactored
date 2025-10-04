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
