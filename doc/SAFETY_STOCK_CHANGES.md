# Safety Stock Refactor — Session Notes

> **Author:** AI-assisted session (Claude)
> **Files touched:** `backend/model/forecaster.py`, `backend/backtest.py`, `backend/utils/plots.py`, `backend/config/config_data_xlsx.yaml`

---

## Context

The safety stock logic was producing unrealistically small values (e.g. 780 units when the
December spike demands ~28,000). This document explains every change made to fix it, in the
order the problems were discovered.

---

## 1. Plots were ignoring the new array-valued safety stock

### Problem
`stock_analysis()` was already returning a point-by-point `safety_stock_test` array, but the
plot function still treated `safety_stock` as a scalar — it only drew a shaded band of fixed
width above the forecast line, indistinguishable from a flat line.

### Fix — `backend/utils/plots.py` (`visualize_forecast_and_safety_stock`)

```python
# Ensure safety_stock is an array aligned with y_pred_prod
safety_stock = np.asarray(safety_stock, dtype=float)
if safety_stock.ndim == 0 or safety_stock.size == 1:
    safety_stock = np.full(len(y_pred_prod), float(safety_stock.flat[0]))

ss_upper = y_pred_prod + safety_stock

# Shade area between forecast and forecast + SS
ax.fill_between(days, y_pred_prod, ss_upper, ...)

# NEW: explicit trigger line so variability is visible
ax.plot(days, ss_upper, ..., label='Safety Stock Trigger')
```

The explicit dotted line `ss_upper = forecast + safety_stock` makes December's wider band
visually obvious instead of hiding it inside the fill.

### Fix — `backend/model/forecaster.py` (`stock_analysis`)

Two calls were still using the scalar mean instead of the array:

```python
# Inventory band chart — was: np.full_like(stock_base, safety_stock)
fig_inventory_band = visualize_inventory_scenarios(
    ...
    safety_stock_line=safety_stock_test,   # ← point-by-point array
)

# Returned DataFrame — was: "safety_stock": safety_stock  (scalar, broadcast to all rows)
return pd.DataFrame({
    ...
    "safety_stock": safety_stock_test,     # ← per-period array
    ...
})
```

---

## 2. Safety stock was computed from training residuals (too optimistic)

### Problem
The `rel_buffer` (the % buffer applied to each forecast) was computed from the 95th percentile
of **in-sample training residuals**. The model had already minimised those errors during
fitting, so they underestimate real production errors — the resulting safety stock was too thin.

### Fix — Option 1: OOF (Out-Of-Fold) residuals from backtest cuts

**Idea:** each backtest cut produces **out-of-sample** predictions vs. actuals. Those errors are
unbiased because the model never trained on them. We accumulate those residuals as cuts run
and pass them into the next cut's safety stock calibration.

#### `backend/backtest.py` — accumulator

```python
# Before the cuts loop
oof_rel_residuals: List[float] = []

for cut_idx, cut in enumerate(cuts, start=1):

    # Snapshot of prior cuts only — cut_1 gets None
    oof_snapshot = np.asarray(oof_rel_residuals) if oof_rel_residuals else None

    # ... fit, predict ...

    # Append this cut's OOS relative residuals AFTER taking the snapshot
    _denom = np.abs(yhat_test) + 1e-6
    oof_rel_residuals.extend(((y_true - yhat_test) / _denom).tolist())

    # Pass prior-cuts snapshot to stock_analysis
    stock_df = fore.stock_analysis(
        ...
        oof_residuals=oof_snapshot,
    )
```

**Key invariant:** cut k only receives residuals from cuts 1..k-1 (never its own test window).
cut_1 always gets `None` and falls back to the training-based estimate.

#### `backend/model/forecaster.py` — mixed pool logic

```python
def stock_analysis(self, ..., oof_residuals: Optional[np.ndarray] = None):

    # Training relative residuals from the last `safety_stock_window` months
    _train_rel = (y_train[-window:] - y_pred_train[-window:]) / (y_pred_train[-window:] + 1e-6)

    # Start with inflated training residuals as a seasonal "prior"
    _rel_for_ss = _train_rel * 1.5   # 1.5× inflates to compensate in-sample optimism

    # Append OOF residuals when available (≥ 3 points)
    if oof_residuals is not None and len(oof_residuals) >= 3:
        _rel_for_ss = np.concatenate([_rel_for_ss, oof_residuals])

    rel_buffer = max(np.quantile(_rel_for_ss, expected_service_level), 0.0)
```

---

## 3. OOF residuals from summer calibrated winter safety stock incorrectly

### Problem
With `n_cuts=2` and `test_len=4m`:
- cut_1 test window = **May–Aug 2024** (flat summer, small errors)
- cut_2 test window = **Sep–Dec 2024** (includes December spike)

cut_2 would use cut_1's 4 summer residuals for safety stock calibration. 95th percentile of
4 small-error months = **6.3%** → SS = 6.3% × 12,000 = 780 units, which is totally
inadequate for December demand of ~28,000.

### Fix — pool OOF + inflated training (not "either-or")

Rather than replacing training residuals with OOF residuals, we **concatenate both**:

```
pool = [train_residuals × 1.5] + [OOF_residuals_from_prior_cuts]
```

Training residuals cover 18 months including December 2023, so the December training error
(model predicted ~10K, actual was ~28K → relative error ≈ +133%) is always in the pool.
The 95th percentile of the combined pool is anchored to December even when OOF data is
all from summer.

As more cuts accumulate over time and cover different seasons, the OOF pool grows and
gradually dominates the training prior.

---

## 4. `safety_stock_window: 2` excluded December from training residuals

### Problem
`safety_stock_window: 2` told the code to use only the **last 2 training months** for the
training residual pool. With boundary=Aug 2024, those 2 months are July + August —
the flattest months of the year. December 2023 (the biggest error) was excluded entirely.

Even with the mixed-pool fix above, if December 2023 is not in the training window,
the seasonal anchor is missing.

### Fix — `backend/config/config_data_xlsx.yaml`

```yaml
stock:
  safety_stock_window: 24   # was: 2 — now covers full training history incl. Dec 2023
  lead_time: 1
  service_level: 0.95
```

With `window=24`, the training residual pool spans the entire 18-month training window,
which always includes at least one December reference. The December training relative
error (≈ 133–200% after inflation) anchors the 95th percentile and produces a
realistic safety stock for the spike months.

---

## Summary of changes

| File | What changed |
|---|---|
| `utils/plots.py` | `visualize_forecast_and_safety_stock` normalises `safety_stock` to array; adds explicit trigger line |
| `model/forecaster.py` | `stock_analysis()` — inventory band and DataFrame now use `safety_stock_test` array; new `oof_residuals` parameter; mixed-pool (OOF + inflated training) quantile |
| `backtest.py` | `oof_rel_residuals` accumulator; snapshot passed to `stock_analysis` per cut |
| `config/config_data_xlsx.yaml` | `safety_stock_window: 2 → 24` |

---

## Why the safety stock was flat before

`safety_stock_test[t] = rel_buffer × forecast[t]`

If `rel_buffer` is a single scalar and `forecast[t]` is approximately flat (the model doesn't
predict the December spike), then `safety_stock_test` is also flat. The variability in safety
stock is **inherited from the variability in the forecast**. Improving the forecast (e.g. by
using ETS or adding a `seasonal_index` feature) will automatically produce a December safety
stock that is 2–3× higher than the surrounding months, without any further formula changes.
