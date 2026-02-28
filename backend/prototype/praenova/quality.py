from __future__ import annotations

from typing import Any

import pandas as pd


def _continuity_report(df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], int]:
    continuity: dict[str, dict[str, Any]] = {}
    missing_total = 0

    for series_id, g in df.groupby("series_id"):
        g = g.sort_values("date")
        start = pd.Timestamp(g["date"].min())
        end = pd.Timestamp(g["date"].max())
        expected = pd.date_range(start=start, end=end, freq="W-MON")
        observed = pd.DatetimeIndex(g["date"])
        missing = expected.difference(observed)
        missing_count = int(len(missing))
        missing_total += missing_count
        continuity[str(series_id)] = {
            "n_obs": int(len(g)),
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "expected_weeks": int(len(expected)),
            "missing_weeks": missing_count,
            "is_monotonic_increasing": bool(observed.is_monotonic_increasing),
        }
    return continuity, missing_total


def run_quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_key_count = int(df.duplicated(subset=["series_id", "date"]).sum())
    continuity_by_series, total_missing_weeks = _continuity_report(df)

    promo_type_present = df["promo_type"].fillna("").astype(str).str.strip().ne("")
    discount_positive = df["promo_discount_pct"] > 0.0

    invalid_week_mask = (df["week_num"] < 1) | (df["week_num"] > 53)

    value_range_flags = {
        "negative_sales_units_count": int((df["sales_units"] < 0).sum()),
        "negative_price_eur_count": int((df["price_eur"] < 0).sum()),
        "negative_sellout_eur_count": int((df["sellout_eur"] < 0).sum()),
        "negative_competitor_price_eur_count": int((df["competitor_price_eur"] < 0).sum()),
        "negative_promo_discount_pct_count": int((df["promo_discount_pct"] < 0).sum()),
        "promo_discount_gt_1_count": int((df["promo_discount_pct"] > 1).sum()),
        "invalid_week_num_count": int(invalid_week_mask.sum()),
    }

    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "series_count": int(df["series_id"].nunique()),
        "series_ids": sorted(df["series_id"].astype(str).unique().tolist()),
        "missing_by_column": {col: int(v) for col, v in df.isna().sum().to_dict().items()},
        "duplicate_key_count": duplicate_key_count,
        "continuity_by_series": continuity_by_series,
        "continuity_issues_count": int(total_missing_weeks),
        "value_range_flags": value_range_flags,
        "promo_consistency_flags": {
            "promo_type_without_discount_count": int((promo_type_present & ~discount_positive).sum()),
            "discount_without_promo_type_count": int((discount_positive & ~promo_type_present).sum()),
            "promo_flag_mismatch_count": int((df["promo_flag"] != (promo_type_present | discount_positive).astype(int)).sum()),
        },
    }
    return report
