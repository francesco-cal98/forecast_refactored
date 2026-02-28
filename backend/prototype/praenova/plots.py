from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ordered_series_ids(df: pd.DataFrame) -> list[str]:
    order_df = (
        df[["series_id", "product", "area"]]
        .drop_duplicates()
        .sort_values(["product", "area"])
    )
    return order_df["series_id"].tolist()


def _make_facet_grid(n_panels: int, ncols: int = 3, figsize_per_col: float = 5.5):
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_col * ncols, 3.2 * nrows),
        sharex=False,
        sharey=True,
    )
    if not isinstance(axes, (list, tuple)):
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    else:
        axes = list(axes)
    axes = axes if isinstance(axes, list) else list(axes)
    return fig, axes, nrows, ncols


def plot_trend_facets(df: pd.DataFrame):
    sns.set_style("whitegrid")
    series_ids = _ordered_series_ids(df)
    fig, axes, nrows, ncols = _make_facet_grid(len(series_ids))
    axes = axes if isinstance(axes, list) else list(axes)

    for idx, series_id in enumerate(series_ids):
        ax = axes[idx]
        g = df[df["series_id"] == series_id].sort_values("date")
        ax.plot(g["date"], g["sales_units"], color="#1f77b4", linewidth=1.8, label="Sales")
        rolling = g["sales_units"].rolling(window=4, min_periods=1).mean()
        ax.plot(g["date"], rolling, color="#d62728", linewidth=1.3, linestyle="--", label="4W MA")
        ax.set_title(series_id, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Sales units")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")

    for idx in range(len(series_ids), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-Series Sales Trend", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_seasonality_facets(df: pd.DataFrame):
    sns.set_style("whitegrid")
    series_ids = _ordered_series_ids(df)
    fig, axes, nrows, ncols = _make_facet_grid(len(series_ids))
    axes = axes if isinstance(axes, list) else list(axes)

    for idx, series_id in enumerate(series_ids):
        ax = axes[idx]
        g = df[df["series_id"] == series_id]
        by_week = (
            g.groupby("week_num")["sales_units"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "sales_mean", "std": "sales_std"})
        )
        by_week["sales_std"] = by_week["sales_std"].fillna(0.0)
        ax.plot(by_week["week_num"], by_week["sales_mean"], color="#2ca02c", linewidth=1.8)
        ax.fill_between(
            by_week["week_num"],
            by_week["sales_mean"] - by_week["sales_std"],
            by_week["sales_mean"] + by_week["sales_std"],
            color="#2ca02c",
            alpha=0.2,
        )
        ax.set_title(series_id, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Sales units")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Week of year")

    for idx in range(len(series_ids), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-Series Seasonality Profile (Week-of-Year)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_promo_impact_facets(df: pd.DataFrame):
    sns.set_style("whitegrid")
    series_ids = _ordered_series_ids(df)
    fig, axes, nrows, ncols = _make_facet_grid(len(series_ids))
    axes = axes if isinstance(axes, list) else list(axes)

    for idx, series_id in enumerate(series_ids):
        ax = axes[idx]
        g = df[df["series_id"] == series_id]
        grp = g.groupby("promo_flag")["sales_units"].mean()
        nonpromo = float(grp.get(0, 0.0))
        promo = float(grp.get(1, 0.0))
        ax.bar(["No promo", "Promo"], [nonpromo, promo], color=["#7f7f7f", "#ff7f0e"])
        ax.set_title(series_id, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.tick_params(axis="y", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Mean sales units")
        if int((g["promo_flag"] == 1).sum()) == 0:
            ax.text(0.5, 0.9, "No promo weeks", transform=ax.transAxes, ha="center", va="center", fontsize=8)

    for idx in range(len(series_ids), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-Series Promo Impact (Mean Sales)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_price_sales_facets(df: pd.DataFrame):
    sns.set_style("whitegrid")
    series_ids = _ordered_series_ids(df)
    fig, axes, nrows, ncols = _make_facet_grid(len(series_ids))
    axes = axes if isinstance(axes, list) else list(axes)

    for idx, series_id in enumerate(series_ids):
        ax = axes[idx]
        g = df[df["series_id"] == series_id]
        sns.scatterplot(
            data=g,
            x="price_eur",
            y="sales_units",
            hue="promo_flag",
            palette={0: "#1f77b4", 1: "#d62728"},
            alpha=0.75,
            s=18,
            linewidth=0,
            ax=ax,
            legend=False,
        )
        ax.set_title(series_id, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Sales units")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Price (EUR)")

    for idx in range(len(series_ids), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Per-Series Price vs Sales (color by Promo Flag)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
