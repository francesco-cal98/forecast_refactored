# Codebase Guide for FastAPI + Frontend Integration

## 1) Purpose
This document is a handoff guide for engineers implementing:
- FastAPI endpoints on top of the forecasting codebase
- Frontend screens that trigger runs and visualize outputs

It describes the current code as-is, including runtime contracts, artifact paths, and important caveats.

---

## 2) High-Level Structure
Main directories:
- `backend/` Forecasting engine (Hydra entrypoint, backtest runner, dataset prep, model wrapper, plots/metrics)
- `backend/config/` Hydra configs for model/data/training/backtest
- `backend/prototype/praenova/` EDA prototype for `praenova_validation_2y.xlsx`
- `src/forecasting/regulatory/` Separate regulatory ingestion/enrichment pipeline
- `conf/regulatory/` Hydra configs for regulatory pipeline
- `scripts/` Utility CLIs (regulatory pipeline orchestration, parquet viewer)
- `tests/` Unit/smoke tests (forecasting + regulatory + praenova prototype)
- `outputs/` and `backend/outputs/` Generated runtime artifacts (plots/csv/xlsx)
- `mlruns/` and `backend/mlruns/` MLflow file-store experiments/runs

Two largely separate products coexist:
- Demand forecasting + inventory simulation
- Regulatory text ingestion/enrichment

---

## 3) Forecasting Runtime Architecture

### 3.1 Main entrypoint
File: `backend/main.py`

Execution modes:
- Backtest mode (`cfg.backtest.enabled=true`, default): loops all products/series and calls `BacktestRunner.run_all(...)`
- Single-run mode (`cfg.backtest.enabled=false`): trains one series and prints WAPE

Core classes wired in main:
- `SalesDataset` from `backend/data/dataset.py`
- `Forecaster` from `backend/model/forecaster.py`
- `BacktestRunner` from `backend/backtest.py`

Hydra:
- default config name is `config` under `backend/config/`
- alternative config for praenova exists: `config_praenova`

Run examples:
- `cd backend`
- `python main.py`
- `python main.py backtest.enabled=false`
- `python main.py --config-name config_praenova`

### 3.2 Data loading and feature engineering
File: `backend/data/dataset.py`

`SalesDataset` responsibilities:
- Load dataset from CSV or XLSX
- Normalize date column
- Build time-series features
- Split train/test around rolling boundary
- Select final columns based on config
- Scale targets per series (median-based scale factor)
- Encode product identifiers

Special loader path:
- If `dataset_config.loader == "praenova"` and input is XLSX, it calls:
  - `backend/prototype/praenova/io.py::load_area_panel(...)`
- That loader uses only area sheets (`Product_XX_AreaY`)

Feature families currently implemented:
- Lags
- Percentage changes
- Rolling stats (`max/min/std/median`)
- Trend slopes
- Fourier seasonal features
- Calendar features (`week_number`, `month`, `quarter`, etc.)
- Market-level cross-series features
- EMA / weeks-since-peak / z-score features
- Price gap/relative price features (when `Prezzo` is available)

### 3.3 Train/test split logic
File: `backend/data/dataset.py::process_data`

Behavior:
- Boundary is selected on target series dates
- Test mask is always target series in `[boundary, boundary + test_len)`
- Train mask depends on `dataset_config.training_mode`
  - `local`: target series history only
  - `global`: target history + all other series

Important caveat:
- Current code creates forward target `y(t+h)` then splits by `t < boundary`
- This can include train rows whose label lands at test start (`h=1` leak)
- Additionally, in `global` mode, other series rows are not date-capped and can include future periods

Implication for API consumers:
- Treat current backtest numbers as operational but not final for governance until split leakage is fixed

### 3.4 Model wrapper
File: `backend/model/forecaster.py`

Responsibilities:
- Initialize model by config (`random_forest`, `gradient_boosting`, `light_gbm`, `svr`)
- Fit with optional temporal sample weights
- Predict and compute error metric
- Perform inventory simulation (`stock_analysis`) and save stock-related plots

Inventory outputs include:
- Safety stock
- Service level
- Lost sales paths
- Stock coverage

### 3.5 Backtesting engine
File: `backend/backtest.py`

`BacktestRunner`:
- Builds rolling-origin cuts per item
- Trains/predicts each cut
- Computes metrics (`MAPE`, `WAPE`, `SMAPE`, `MASE`)
- Computes probabilistic diagnostics (`P10/P50/P90` coverage and horizon error)
- Runs stock analysis per cut
- Logs params/metrics/artifacts to MLflow
- Writes per-cut and per-item artifacts to disk

---

## 4) Forecasting Artifacts Contract (for API/Frontend)

### 4.1 Output folder pattern
Configured by `cfg.backtest.output_dir`:
- Default standard config: `outputs/backtest`
- Praenova config: `outputs/backtest_praenova`

Per item/cut:
- `{output_dir}/{item}/cut_{k}/preds_{item}_cut{k}.csv`
- `{output_dir}/{item}/cut_{k}/stock_analysis_{item}_cut{k}.xlsx` (or CSV fallback)
- `{output_dir}/{item}/cut_{k}/{item}_stock_analysis.png`
- `{output_dir}/{item}/cut_{k}/{item}_stock_analysis_train.png`
- `{output_dir}/{item}/cut_{k}/{item}_inventory_scenario_band.png`
- `{output_dir}/{item}/cut_{k}/{item}_forecast_fanchart.png`
- Optional feature importance image if model supports it

Per item aggregate:
- `{output_dir}/preds_{item}.csv`
- `{output_dir}/{item}/prob_diagnostics_{item}.csv`
- `{output_dir}/{item}/{item}_error_vs_horizon.png`
- `{output_dir}/{item}/{item}_coverage_10_90.png` (if available)

### 4.2 Data columns used in frontend plots
From prediction CSV rows:
- `item`
- `cut_idx`
- `h`
- `timestamp`
- `y_true`
- `y_pred` (alias of p50)
- `p50`
- `p10`
- `p90`

From stock analysis table:
- `y_test`, `y_pred_test`
- `stock_test`, `forecast_sell_in_test`, `lost_sales`
- `stock_low`, `stock_base`, `stock_high`
- `safety_stock`
- `service_level`, `service_level_low`, `service_level_base`, `service_level_high`
- `stock_cover`

---

## 5) Config System (Hydra)

### 5.1 Standard config
File: `backend/config/config.yaml`

Defaults:
- model: `light_gbm`
- data: `xlsx_sample`

Groups:
- `backend/config/data/*.yaml`
- `backend/config/model/*.yaml`

Main sections:
- `dataset_config`
- `prepare`
- `column_selection`
- `train`
- `cv`
- `stock`
- `mlflow`
- `backtest`

### 5.2 Praenova config
Files:
- `backend/config/config_praenova.yaml`
- `backend/config/data/praenova.yaml`

Praenova specifics:
- data source is `praenova_validation_2y.xlsx`
- target column is `sales_units`
- product key is `series_id` (`Product_XX_AreaY`)
- `dataset_config.loader: praenova`
- `dataset_config.training_mode: local`
- `backtest.output_dir: outputs/backtest_praenova`

---

## 6) Praenova EDA Prototype

Location: `backend/prototype/praenova/`

Modules:
- `io.py` load area-level panel from xlsx sheets
- `quality.py` quality checks
- `summary.py` per-series summary table
- `plots.py` multi-facet dashboards
- `run_eda.py` CLI to generate all EDA artifacts

Notebook:
- `notebooks/praenova_validation_2y_eda.ipynb`
- Uses robust project-root detection for imports

EDA output folder:
- `outputs/eda/praenova_validation_2y/`

---

## 7) Regulatory Pipeline (Separate Product)

Entrypoints:
- `src/forecasting/regulatory/ingest/run_ingest.py`
- `src/forecasting/regulatory/enrichment/run_enrichment.py`

Config root:
- `conf/regulatory/*`

Orchestration script:
- `scripts/pipeline_runner.py`

Do not mix this pipeline with forecasting API endpoints unless explicitly needed.

---

## 8) Current FastAPI Status

Important:
- `fastapi` and `uvicorn` are listed in `requirments.txt`
- No production FastAPI app module currently exists in the repo
- Integration work still needs to create an API layer

Recommended API placement:
- `backend/api/app.py` (FastAPI application)
- `backend/api/schemas.py` (Pydantic request/response models)
- `backend/api/services/forecast_service.py` (pure orchestration around current classes)
- `backend/api/services/artifact_service.py` (artifact discovery + URL mapping)

---

## 9) Suggested FastAPI Endpoint Contract

### 9.1 Health
- `GET /health`
- response: `{ "status": "ok" }`

### 9.2 Config discovery
- `GET /configs`
- response: available model/data/config names

### 9.3 Run single forecast
- `POST /forecast/run`
- request example:
  - `config_name`: `config` or `config_praenova`
  - overrides: list of hydra-style overrides
  - target item/series
- response:
  - key metrics
  - prediction vectors
  - artifact paths/URLs

### 9.4 Run backtest job
- `POST /backtest/run`
- should be async job-oriented for long runs
- response:
  - `job_id`
  - status endpoint URL

### 9.5 Backtest status and result
- `GET /backtest/{job_id}`
- response:
  - state (`queued`, `running`, `finished`, `failed`)
  - logs
  - summary metrics
  - artifact manifest

### 9.6 Artifacts
- `GET /artifacts/{job_id}` list files
- `GET /artifacts/{job_id}/file?...` stream/download CSV/PNG/XLSX

---

## 10) Suggested Frontend Screens

Minimum UI:
- Run form (config, model, dataset, n_cuts, item filter)
- Job status panel (live logs + progress)
- Metrics summary cards (`MAPE`, `WAPE`, `SMAPE`, `MASE`, service level)
- Prediction chart (`y_true`, `p50`, `p10`, `p90`)
- Inventory charts (stock, safety stock, sell-in, stock cover)
- Artifact table with download links

For praenova:
- Series selector based on `series_id` values
- Optional dashboard views from EDA output PNGs

---

## 11) Command Runbook

From repo root:
- `cd backend`

Standard backtest:
- `python main.py backtest.enabled=true hydra.job.chdir=False`

Praenova backtest:
- `python main.py --config-name config_praenova backtest.enabled=true hydra.job.chdir=False`

Single run (standard):
- `python main.py backtest.enabled=false hydra.job.chdir=False`

Single run (praenova):
- `python main.py --config-name config_praenova backtest.enabled=false hydra.job.chdir=False`

Praenova EDA:
- `python -m backend.prototype.praenova.run_eda --input praenova_validation_2y.xlsx --output outputs/eda/praenova_validation_2y --base-year 2023`

Tests:
- `python -m pytest`
- targeted praenova tests:
  - `python -m pytest tests/test_praenova_io.py tests/test_praenova_quality.py tests/test_praenova_summary.py tests/test_praenova_plots.py tests/test_praenova_runner.py`

---

## 12) Known Caveats and Risks

1. Split leakage risk exists in current `SalesDataset.process_data` logic for forward targets (`nr_horizon >= 1`).
2. `global` training mode can include future rows from non-target series unless date-capped.
3. `config_praenova.yaml` misses Hydra `_self_` default ordering marker warning.
4. MLflow currently uses file store (`file:./mlruns`), which is fine for local dev but not ideal for multi-user prod.
5. Some legacy tests are stale against current interfaces (especially older dataset/stock-analysis expectations).
6. Paths in configs use absolute Windows paths; production/API deployment should convert to environment-driven paths.

---

## 13) Integration Checklist for Your Friend

1. Create FastAPI app skeleton under `backend/api/`.
2. Wrap existing classes (`SalesDataset`, `Forecaster`, `BacktestRunner`) in service functions, do not duplicate logic.
3. Add async job execution for backtests (thread/process queue).
4. Standardize artifact manifest JSON so frontend has stable URLs.
5. Expose config discovery endpoint to avoid hardcoded UI options.
6. Add request validation around config overrides and target item names.
7. Add auth/rate-limiting/logging if external users are expected.
8. Add one end-to-end API test that runs a tiny one-item one-cut backtest.

---

## 14) Quick “Who Owns What” Map

- Forecasting orchestration: `backend/main.py`, `backend/backtest.py`
- Dataset prep/splits/features: `backend/data/dataset.py`
- Model fit/predict/inventory simulation: `backend/model/forecaster.py`
- Metrics and forecast plots: `backend/utils/metrics.py`, `backend/utils/plots.py`
- Hydra configs: `backend/config/*`
- Praenova EDA/prototype ingestion: `backend/prototype/praenova/*`
- Regulatory ingestion/enrichment: `src/forecasting/regulatory/*`, `conf/regulatory/*`

---

## 15) Recommended Immediate Next Step
Before exposing metrics in frontend, patch split leakage in `SalesDataset.process_data` and re-baseline backtest metrics.

This avoids shipping a UI/API over potentially optimistic validation numbers.

