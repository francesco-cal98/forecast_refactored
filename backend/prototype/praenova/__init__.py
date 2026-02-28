"""Praenova per-series EDA prototype package."""

from .io import load_area_panel
from .quality import run_quality_checks
from .summary import build_series_summary
from .plots import (
    plot_trend_facets,
    plot_seasonality_facets,
    plot_promo_impact_facets,
    plot_price_sales_facets,
)

__all__ = [
    "load_area_panel",
    "run_quality_checks",
    "build_series_summary",
    "plot_trend_facets",
    "plot_seasonality_facets",
    "plot_promo_impact_facets",
    "plot_price_sales_facets",
]
