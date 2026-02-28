from __future__ import annotations

from datetime import date
import re
from pathlib import Path

import pandas as pd


AREA_SHEET_RE = re.compile(r"^(Product_\d+)_(Area\d+)$")

INPUT_COLUMNS = {
    "Year": "year_num",
    "Week": "week_num",
    "Sales (units)": "sales_units",
    "Price (€)": "price_eur",
    "Promotion Type": "promo_type",
    "Promotion Discount %": "promo_discount_pct",
    "Sell-Out (€)": "sellout_eur",
    "Trend Index": "trend_index",
    "Competitor Price (€)": "competitor_price_eur",
}

OUTPUT_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "year_num",
    "week_num",
    "sales_units",
    "price_eur",
    "promo_discount_pct",
    "sellout_eur",
    "trend_index",
    "competitor_price_eur",
]


def _parse_area_sheet_name(sheet_name: str) -> tuple[str, str] | None:
    match = AREA_SHEET_RE.match(sheet_name)
    if not match:
        return None
    product, area = match.groups()
    return product, area


def _iso_monday_from_year_week(base_year: int, year_num: int, week_num: int) -> pd.Timestamp:
    iso_year = int(base_year) + int(year_num) - 1
    try:
        dt = date.fromisocalendar(iso_year, int(week_num), 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ISO year/week combination: base_year={base_year}, "
            f"year_num={year_num}, week_num={week_num}, iso_year={iso_year}"
        ) from exc
    return pd.Timestamp(dt)


def _validate_required_columns(df: pd.DataFrame, sheet_name: str) -> None:
    missing = sorted(set(INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' is missing required columns: {missing}")


def load_area_panel(path: str, base_year: int = 2023) -> pd.DataFrame:
    """Load only Product_xx_AreaY sheets and return a normalized panel dataframe."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    xls = pd.ExcelFile(source_path)
    frames: list[pd.DataFrame] = []

    for sheet_name in xls.sheet_names:
        parsed = _parse_area_sheet_name(sheet_name)
        if parsed is None:
            continue
        product, area = parsed
        raw = xls.parse(sheet_name)
        _validate_required_columns(raw, sheet_name)

        frame = raw[list(INPUT_COLUMNS.keys())].rename(columns=INPUT_COLUMNS).copy()
        frame["product"] = product
        frame["area"] = area
        frame["series_id"] = f"{product}_{area}"

        for col in NUMERIC_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if frame[col].isna().any():
                count = int(frame[col].isna().sum())
                raise ValueError(
                    f"Column '{col}' in sheet '{sheet_name}' contains {count} non-numeric or null values."
                )

        frame["year_num"] = frame["year_num"].astype(int)
        frame["week_num"] = frame["week_num"].astype(int)
        frame["date"] = frame.apply(
            lambda row: _iso_monday_from_year_week(
                base_year=base_year,
                year_num=int(row["year_num"]),
                week_num=int(row["week_num"]),
            ),
            axis=1,
        )

        frame["promo_type"] = frame["promo_type"].astype("string")
        promo_type_present = frame["promo_type"].fillna("").str.strip().ne("")
        frame["promo_flag"] = (
            promo_type_present | (frame["promo_discount_pct"].astype(float) > 0.0)
        ).astype(int)

        frames.append(frame)

    if not frames:
        raise ValueError(
            "No area sheets found. Expected sheet names matching '^Product_\\d+_Area\\d+$'."
        )

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out[OUTPUT_COLUMNS].sort_values(["product", "area", "date"]).reset_index(drop=True)

    duplicated_keys = out.duplicated(subset=["series_id", "date"]).sum()
    if duplicated_keys > 0:
        raise ValueError(f"Found {int(duplicated_keys)} duplicated (series_id, date) keys.")

    return out
