"""
feature_experiment.py — Forward feature selection experiment
============================================================
Starts with the minimal feature set, adds one feature group at a time,
and records WAPE + plots for each configuration so you can see whether
adding features actually helps.

Each run creates a folder under outputs/feature_experiments/:
  exp_01_calendar/
    features.txt          ← feature list + per-product WAPE + pooled WAPE
    Parfum_1_forecast.png
    Parfum_2_forecast.png
    feature_importance.png  (if model supports it)

Usage:
  cd backend
  python feature_experiment.py --config-name=config_data_xlsx
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

from data.dataset import SalesDataset
from model.forecaster import Forecaster
from utils.helper import resolve_target_keys
from utils.metrics import WAPE
from utils.plots import plot_feat_importance

# ──────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# Each tuple: (experiment_name, list_of_regex_patterns_to_ADD_at_this_step)
# Patterns are cumulative: step N includes all patterns from steps 1..N.
# Use None as the patterns list to include ALL remaining columns (full set).
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_STEPS: List[Tuple[str, Optional[List[str]]]] = [
    (
        "01_calendar",
        # Pure calendar signal — no lag, no rolling, no Fourier
        [r"^month$", r"^quarter$", r"^target_month$", r"^target_quarter$"],
    ),
    (
        "02_add_fourier_k3",
        # Add the single best Fourier harmonic (yearly cycle, k=3)
        # Both source-date and target-date versions
        [r"fourier_(sin|cos)_k3$", r"target_fourier_(sin|cos)_k3$"],
    ),
    (
        "03_add_rolling_std6",
        # Add rolling volatility at 6-month window (drop 3M — correlated)
        [r"rolling_std_6M$"],
    ),
    (
        "04_add_rolling_median12",
        # Add long-term level estimate (12-month rolling median)
        [r"rolling_median_12M$"],
    ),
    (
        "05_add_rolling_min",
        # Add rolling minimum 3M (tracks demand floor)
        [r"rolling_min_3M$"],
    ),
    (
        "06_add_all_fourier",
        # Add remaining Fourier harmonics k1, k2 (both source + target date)
        [r"fourier_(sin|cos)_k[12]$", r"target_fourier_(sin|cos)_k[12]$"],
    ),
    (
        "07_add_shorter_windows",
        # Add shorter rolling windows (more correlated with 6M versions)
        [r"rolling_std_3M$", r"rolling_min_6M$", r"rolling_median_[36]M$"],
    ),
    (
        "08_full",
        # All remaining features
        None,
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _filter_columns(df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """Keep only numeric columns whose name matches ANY of the regex patterns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    kept = [c for c in numeric_cols if any(re.search(p, c) for p in patterns)]
    return df[kept]


def _parse_td(s: str) -> pd.Timedelta:
    s = str(s).strip().lower()
    if s.endswith("w"):
        return pd.Timedelta(weeks=int(s[:-1]))
    if s.endswith("m"):
        return pd.Timedelta(days=30 * int(s[:-1]))
    return pd.to_timedelta(s)


def _build_cuts(dates: pd.Series, test_len_str: str, n_cuts: int):
    test_len = _parse_td(test_len_str)
    t_max = pd.to_datetime(dates.max())
    return [t_max - i * test_len for i in range(n_cuts, 0, -1)]


def _plot_forecast(item: str, y_true, y_pred, cut_idx: int, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(1, len(y_true) + 1)
    ax.plot(x, y_true, label="Actual", linewidth=2, color="#2F6690")
    ax.plot(x, y_pred, label="Forecast", linewidth=2, linestyle="--", color="#F26419")
    ax.set_title(f"{item} — cut {cut_idx}  (WAPE={WAPE(y_true, y_pred):.1f}%)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Demand")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fp = out_dir / f"{item}_cut{cut_idx}_forecast.png"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)


def _run_experiment(
    exp_name: str,
    col_patterns: Optional[List[str]],
    items: List[str],
    cfg: DictConfig,
    prepare_config,
    base_out: Path,
) -> dict:
    """Run one experiment configuration, return summary dict."""
    out_dir = base_out / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.data
    dataset_extra_cfg = getattr(cfg, "dataset_config", {})

    abs_error_sum = 0.0
    abs_actual_sum = 0.0
    per_product = {}

    all_feature_names = None  # captured from first product/cut
    all_importances   = None

    for item in items:
        # Probe available dates
        ds_probe = SalesDataset(
            csv_file=dataset_cfg.file_path,
            separator=dataset_cfg.separator,
            date_column=dataset_cfg.date_column,
            target_column=dataset_cfg.target_column,
            nr_horizon=dataset_cfg.nr_horizon,
            prepare_config=prepare_config,
            column_selection=cfg.column_selection,
            product_var=dataset_cfg.product_column,
            product_name=item,
            granularity=dataset_cfg.granularity,
            config=dataset_extra_cfg,
        )
        df_full = ds_probe._prepare_dataset().sort_values(
            [ds_probe.product_var, ds_probe.date_column]
        )
        df_full = ds_probe._encode_time(df_full, time_unit=ds_probe.granularity)
        tgt_dates = df_full.loc[
            df_full[ds_probe.product_var] == item, ds_probe.date_column
        ]
        boundaries = _build_cuts(tgt_dates, cfg.train.test_len, cfg.backtest.n_cuts)

        item_abs_err = 0.0
        item_abs_act = 0.0

        for cut_idx, boundary in enumerate(boundaries, start=1):
            ds = SalesDataset(
                csv_file=dataset_cfg.file_path,
                separator=dataset_cfg.separator,
                date_column=dataset_cfg.date_column,
                target_column=dataset_cfg.target_column,
                nr_horizon=dataset_cfg.nr_horizon,
                prepare_config=prepare_config,
                column_selection=cfg.column_selection,
                product_var=dataset_cfg.product_column,
                product_name=item,
                granularity=dataset_cfg.granularity,
                config=dataset_extra_cfg,
            )
            ds.process_data(
                product=item,
                train_params={
                    "train_len": cfg.train.train_len,
                    "test_len": cfg.train.test_len,
                    "boundary": str(boundary.date()),
                },
            )

            _scale = getattr(ds, "series_scale", 1.0)

            # Feature filtering: keep only matched columns (or all if None)
            if col_patterns is not None:
                X_tr = _filter_columns(ds.X_train, col_patterns)
                X_te = _filter_columns(ds.X_test, col_patterns)
            else:
                X_tr = ds.X_train.select_dtypes(include=[np.number])
                X_te = ds.X_test.select_dtypes(include=[np.number])

            if X_tr.empty or X_te.empty:
                print(f"  [{exp_name}] {item}/cut{cut_idx}: no features matched — skipping")
                continue

            fore = Forecaster(
                model_name=cfg.model.name,
                params=cfg.model.params,
            )
            fore.fit(
                X_tr,
                ds.y_train,
                dates=getattr(ds, "train_dates", None),
                recency_gamma=float(getattr(cfg.cv, "recency_gamma", 0.08)),
            )
            y_pred, _ = fore.predict(X_te, ds.y_test)
            y_pred = y_pred * _scale
            y_true = np.asarray(ds.y_test) * _scale

            _plot_forecast(item, y_true, y_pred, cut_idx, out_dir)

            item_abs_err += float(np.sum(np.abs(y_true - y_pred)))
            item_abs_act += float(np.sum(np.abs(y_true)))

            # Capture feature importances from the last cut of the first product
            if (
                all_feature_names is None
                and hasattr(fore.model, "feature_importances_")
                and fore.regressor_names is not None
            ):
                all_feature_names = list(fore.regressor_names)
                all_importances   = fore.model.feature_importances_.copy()

        item_wape = (
            100.0 * item_abs_err / item_abs_act if item_abs_act > 0 else float("nan")
        )
        per_product[item] = round(item_wape, 2)
        abs_error_sum += item_abs_err
        abs_actual_sum += item_abs_act

    pooled_wape = (
        100.0 * abs_error_sum / abs_actual_sum
        if abs_actual_sum > 0
        else float("nan")
    )

    # Feature importance plot
    if all_feature_names and all_importances is not None:
        try:
            plot_feat_importance(
                importance=all_importances,
                feature_names=all_feature_names,
                save_path=str(out_dir / "feature_importance.png"),
            )
        except Exception as e:
            print(f"  [feature importance] {e}")

    # features.txt
    feature_list = (
        sorted(all_feature_names) if all_feature_names
        else ["(patterns matched but names not captured)"]
    )
    n_feat = len(feature_list)
    lines = [
        f"Experiment: {exp_name}",
        f"Model:      {cfg.model.name}",
        f"Features:   {n_feat}",
        "",
        f"Pooled WAPE: {pooled_wape:.2f}%",
        "",
        "Per-product WAPE:",
    ]
    for prod, w in per_product.items():
        lines.append(f"  {prod}: {w:.2f}%")
    lines += ["", "Feature list:"]
    lines += [f"  {f}" for f in feature_list]

    (out_dir / "features.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [{exp_name}]  n_features={n_feat}  pooled_WAPE={pooled_wape:.2f}%")

    return {
        "experiment": exp_name,
        "n_features": n_feat,
        "pooled_wape": round(pooled_wape, 2),
        "per_product": per_product,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    prepare_config = resolve_target_keys(cfg)
    dataset_cfg    = cfg.data
    dataset_extra_cfg = getattr(cfg, "dataset_config", {})

    # Get all product names
    ds_all = SalesDataset(
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
    items = list(ds_all.get_all_products())
    print(f"Products: {items}")

    base_out = Path("outputs/feature_experiments")
    base_out.mkdir(parents=True, exist_ok=True)

    # Build cumulative patterns incrementally
    cumulative_patterns: List[str] = []
    summary_rows = []

    for step_name, new_patterns in FEATURE_STEPS:
        if new_patterns is None:
            # Full set — pass None to use all columns
            active_patterns = None
        else:
            cumulative_patterns = cumulative_patterns + new_patterns
            active_patterns = list(cumulative_patterns)

        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        if active_patterns:
            print(f"  Active patterns: {active_patterns}")
        else:
            print("  Using ALL features")

        result = _run_experiment(
            exp_name=step_name,
            col_patterns=active_patterns,
            items=items,
            cfg=cfg,
            prepare_config=prepare_config,
            base_out=base_out,
        )
        summary_rows.append(result)

    # Write global summary table
    summary_df = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(base_out / "experiment_summary.csv", index=False)

    # Also write a readable txt summary
    lines = ["Feature Selection Experiment Summary", "=" * 50, ""]
    for row in summary_rows:
        lines.append(
            f"{row['experiment']:<35}  n={row['n_features']:<4}  "
            f"WAPE={row['pooled_wape']:.2f}%"
        )
    (base_out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved to: {base_out.resolve()}")


if __name__ == "__main__":
    main()
