# backend/main.py

import hydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import SalesDataset
from model.forecaster import Forecaster
from utils.metrics import MAPE
from utils.helper import resolve_target_keys
from utils.logging_utils import setup_logger

logger = setup_logger("main")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ✅ Risolvi interpolazioni come ${target_column} anche nelle chiavi
    #prepare_config = resolve_config_refs(cfg.prepare, cfg)
    #column_selection = resolve_config_refs(cfg.column_selection, cfg)
    prepare_config = resolve_target_keys(cfg)

    # 1. Load dataset
    dataset = SalesDataset(
        csv_file=cfg.data.file_path,
        separator=cfg.data.separator,
        date_column=cfg.data.date_column,
        target_column=cfg.data.target_column,
        nr_horizon=cfg.data.nr_horizon,
        prepare_config=prepare_config,
        column_selection=cfg.column_selection,
        product_var=cfg.data.product_column,
        product_name=cfg.data.product_name,
        granularity=cfg.data.granularity
    )

    print("✅ Chiavi prepare_config['lag']:", list(prepare_config['lag'].keys()))

    # 2. Process data
    dataset.process_data(
        product=cfg.data.product_name,
        train_params={
            "train_len": cfg.train.train_len,
            "test_len": cfg.train.test_len,
            "boundary": cfg.train.boundary,
        }
    )

    # 3. Train model
    forecaster = Forecaster(model_name=cfg.model.name, params=cfg.model.params)
    forecaster.gridsearch(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=None,             # <— passo X_test per validare solo su target
        param_grid=cfg.model.param_grid,
        scoring='neg_mean_absolute_error' # o la metrica che preferisci
    )

    # 4. Predict
    y_pred, mape = forecaster.predict(dataset.X_test, dataset.y_test, error_fun=MAPE)
    
    

    # 5. Stock analysisd
    df= forecaster.stock_analysis(
        y_test=dataset.y_test,
        y_pred_test=y_pred,
        y_train=dataset.y_train,
        y_pred_train=forecaster.model.predict(dataset.X_train),
        sell_in_true=dataset.X_test.get("orders_SellIn", dataset.y_test),
        safety_stock_time=cfg.stock.safety_stock_window,
        lead_time=cfg.stock.lead_time,
        expected_service_level=cfg.stock.service_level
    )
    df.to_excel(r"C:\Users\fraca\forecast_refactored\backend\df_out.xlsx", index=False)

if __name__ == "__main__":
    main()
