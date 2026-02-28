from pathlib import Path

import pandas as pd
import pytest

from backend.prototype.praenova.io import load_area_panel


def _dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "praenova_validation_2y.xlsx"


def test_load_area_panel_shape_and_schema():
    df = load_area_panel(str(_dataset_path()))
    required = {
        "series_id",
        "product",
        "area",
        "year_num",
        "week_num",
        "date",
        "sales_units",
        "price_eur",
        "promo_type",
        "promo_discount_pct",
        "sellout_eur",
        "trend_index",
        "competitor_price_eur",
        "promo_flag",
    }

    assert df["series_id"].nunique() == 15
    assert len(df) == 1560
    assert required.issubset(set(df.columns))
    assert not df.duplicated(subset=["series_id", "date"]).any()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert set(df["promo_flag"].unique()).issubset({0, 1})


def test_load_area_panel_fails_on_missing_required_column(tmp_path: Path):
    bad = pd.DataFrame(
        {
            "Year": [1, 1],
            "Week": [1, 2],
            "Sales (units)": [10, 12],
            "Price (€)": [1.0, 1.0],
            "Promotion Type": [None, None],
            "Promotion Discount %": [0.0, 0.0],
            "Sell-Out (€)": [10.0, 12.0],
            "Competitor Price (€)": [1.2, 1.1],
        }
    )
    fp = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(fp, engine="openpyxl") as writer:
        bad.to_excel(writer, index=False, sheet_name="Product_99_Area1")

    with pytest.raises(ValueError, match="missing required columns"):
        load_area_panel(str(fp))
