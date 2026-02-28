from pathlib import Path

from backend.prototype.praenova.io import load_area_panel
from backend.prototype.praenova.summary import build_series_summary


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "praenova_validation_2y.xlsx"


def test_build_series_summary_shape_and_columns():
    df = load_area_panel(str(_dataset_path()))
    summary = build_series_summary(df)

    expected_cols = {
        "series_id",
        "product",
        "area",
        "n_obs",
        "start_date",
        "end_date",
        "sales_mean",
        "sales_std",
        "sales_cv",
        "sales_zero_rate",
        "promo_share",
        "price_mean",
        "corr_sales_price",
        "corr_sales_competitor_price",
    }
    assert len(summary) == 15
    assert expected_cols.issubset(set(summary.columns))
    assert (summary["n_obs"] == 104).all()

    critical = ["series_id", "n_obs", "sales_mean", "sales_std", "promo_share", "price_mean"]
    assert summary[critical].notna().all().all()


def test_build_series_summary_stable_order():
    df = load_area_panel(str(_dataset_path()))
    summary = build_series_summary(df)
    assert summary.iloc[0]["series_id"] == "Product_01_Area1"
    assert summary.iloc[-1]["series_id"] == "Product_05_Area3"
