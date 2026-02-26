# Forecasting Engine — SOTA Improvement Roadmap

Scenario: weekly, short multi-product retail panel, supply chain output.
Baseline model: LightGBM (local regime, no ensemble, no scaling).
All expected gains are relative WAPE improvements vs baseline.

---

## ✅ DONE

| # | Technique | Expected WAPE gain | Files changed | Commit |
|---|---|---|---|---|
| 1 | **Fix sample_weight bug** — temporal exponential-decay weights were computed but passed as `None`. Fixed in `forecaster.py` and `backtest.py`. | 1–3% | `forecaster.py`, `backtest.py` | `3b3a0a9` |
| 2 | **Tweedie loss on LGBM** — replaced `quantile(α=0.3)` with `tweedie(power=1.5)`. Handles zero-inflated right-skewed retail demand natively. M5 top-3 loss. | 2–5% | `config/model/light_lgm.yaml` | `3b3a0a9` |
| 3 | **Fourier seasonality features (K=3)** — 3 harmonics (annual, semi-annual, quarterly) replace single sin/cos cycle. Also added `month` and `quarter`. | 3–6% | `data/dataset.py`, `config/config.yaml` | `3b3a0a9` |
| 4 | **Seasonal lag anchors** — lag list extended to `[1,2,3,4,5,26,52]`. lag-26 = 6-month, lag-52 = same-week-last-year. YoY pct_change(52) added. | 2–5% | `config/config.yaml` | `3b3a0a9` |
| 5 | **Per-series mean scaling** — `y_train` and `y_test` divided by median demand of target series in training window. Inverse-transform applied before metrics and stock analysis. | 2–4% | `data/dataset.py`, `backtest.py` | `3b3a0a9` |

---

## 🔲 TODO

| # | Technique | Expected WAPE gain | Effort | Notes |
|---|---|---|---|---|
| 6 | **Fix train regime bug** — other products contribute rows with dates past the boundary. Apply same date cutoff to ALL series when `regime=global`. `process_data()` train_mask fix. | 1–3% | 30 min | `data/dataset.py` process_data() train_mask |
| 7 | **Intermittency routing** — detect sparsity (% zero weeks) at training time. Route series with sparsity > 0.4 to Croston/SBA instead of GBDT. Add `sparsity` feature for all series. | 3–8% on sparse SKUs | 2–3h | New class `CrostonForecaster`. Config flag `intermittency_threshold`. |
| 8 | **Multi-horizon direct forecasting (h=1..8)** — train one independent model per horizon. Each model targets `Sellout_fwd_hW`. Lag features for model_h must be ≥ h to avoid leakage. Enables 8-week planning horizon for supply chain. | Enables new use cases | 4–6h | Loop in `backtest.py`; new config key `horizons: [1,2,3,4,5,6,7,8]` |
| 9 | **Conformal prediction intervals** — calibrate P10/P50/P90 intervals from backtest residuals. Replace Gaussian safety stock formula with empirical P90 quantile. | Better safety stock | 2h | Post-processing step in `backtest.py`. No model changes needed. |
| 10 | **Walk-forward stacking (meta-learner)** — train LGBM + RF per fold, collect OOF predictions, train Ridge meta-learner to optimally blend. ~2× training time. | 2–4% | 3–4h | New `EnsembleForecaster` class. Requires multi-model config. |

---

## Architecture notes

### Training regimes (not yet implemented as config switch)
- **local**: only target product rows in training. Cross-series info via lagged market features only. Use when products are heterogeneous.
- **global**: all products, same date cutoff. Use when many similar products, sparse histories.
- **hybrid**: global pretrain + Ridge residual corrector on recent local window. M5 SOTA. Use for shared seasonality + different product levels.

### Key invariants to preserve
- Walk-forward validation must never shuffle time. StratifiedKFold is invalid for this data.
- Market features (`market_sum_ex`, `market_mean_ex`) are already correctly lagged and exclude the target row's own contribution — do not modify this.
- `series_scale` is set per-product per-cut. Never share scales across products or cuts.
- Lag features for horizon h must satisfy: min_lag >= h. Config currently targets h=1.

---

## Running the backtest

```bash
cd backend
python main.py model=light_lgm backtest.enabled=true backtest.n_cuts=3
```

Override regime or horizon from CLI:
```bash
python main.py model=light_lgm train.regime=global
```
