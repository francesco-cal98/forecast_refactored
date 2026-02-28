from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from backend.prototype.praenova.io import load_area_panel
from backend.prototype.praenova.plots import (
    plot_price_sales_facets,
    plot_promo_impact_facets,
    plot_seasonality_facets,
    plot_trend_facets,
)


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "praenova_validation_2y.xlsx"


def test_plot_functions_save_files(tmp_path: Path):
    df = load_area_panel(str(_dataset_path()))

    figures = {
        "trend.png": plot_trend_facets(df),
        "seasonality.png": plot_seasonality_facets(df),
        "promo.png": plot_promo_impact_facets(df),
        "price_vs_sales.png": plot_price_sales_facets(df),
    }

    for name, fig in figures.items():
        out = tmp_path / name
        fig.savefig(out)
        assert out.exists()


def test_plot_promo_impact_handles_no_promo(tmp_path: Path):
    df = load_area_panel(str(_dataset_path()))
    df_no_promo = df.copy()
    df_no_promo["promo_flag"] = 0
    df_no_promo["promo_type"] = None
    df_no_promo["promo_discount_pct"] = 0.0

    fig = plot_promo_impact_facets(df_no_promo)
    out = tmp_path / "promo_no_promo.png"
    fig.savefig(out)
    assert out.exists()
