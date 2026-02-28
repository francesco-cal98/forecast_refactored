from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if float(a.std(ddof=0)) == 0.0 or float(b.std(ddof=0)) == 0.0:
        return float("nan")
    return float(a.corr(b))


def build_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for series_id, g in df.groupby("series_id", sort=True):
        g = g.sort_values("date")
        sales = g["sales_units"].astype(float)
        prices = g["price_eur"].astype(float)
        competitor_prices = g["competitor_price_eur"].astype(float)
        trend = g["trend_index"].astype(float)

        sales_mean = float(sales.mean())
        sales_std = float(sales.std(ddof=0))
        sales_cv = float(sales_std / sales_mean) if sales_mean != 0.0 else float("nan")

        promo_mask = g["promo_flag"].astype(int) == 1
        promo_sales_mean = float(sales[promo_mask].mean()) if promo_mask.any() else float("nan")
        nonpromo_sales_mean = float(sales[~promo_mask].mean()) if (~promo_mask).any() else float("nan")
        promo_lift_units = (
            float(promo_sales_mean - nonpromo_sales_mean)
            if np.isfinite(promo_sales_mean) and np.isfinite(nonpromo_sales_mean)
            else float("nan")
        )

        weekly_means = g.groupby("week_num")["sales_units"].mean()
        seasonality_strength = (
            float(weekly_means.std(ddof=0) / sales_mean) if sales_mean != 0.0 else float("nan")
        )

        rows.append(
            {
                "series_id": str(series_id),
                "product": str(g["product"].iloc[0]),
                "area": str(g["area"].iloc[0]),
                "n_obs": int(len(g)),
                "start_date": str(pd.Timestamp(g["date"].min()).date()),
                "end_date": str(pd.Timestamp(g["date"].max()).date()),
                "sales_mean": sales_mean,
                "sales_std": sales_std,
                "sales_cv": sales_cv,
                "sales_min": float(sales.min()),
                "sales_p25": float(sales.quantile(0.25)),
                "sales_p50": float(sales.quantile(0.5)),
                "sales_p75": float(sales.quantile(0.75)),
                "sales_max": float(sales.max()),
                "sales_zero_rate": float((sales == 0).mean()),
                "promo_share": float(g["promo_flag"].mean()),
                "promo_lift_units": promo_lift_units,
                "price_mean": float(prices.mean()),
                "price_std": float(prices.std(ddof=0)),
                "price_min": float(prices.min()),
                "price_max": float(prices.max()),
                "discount_mean": float(g["promo_discount_pct"].mean()),
                "discount_positive_rate": float((g["promo_discount_pct"] > 0).mean()),
                "competitor_price_mean": float(competitor_prices.mean()),
                "corr_sales_price": _safe_corr(sales, prices),
                "corr_sales_competitor_price": _safe_corr(sales, competitor_prices),
                "trend_index_mean": float(trend.mean()),
                "trend_index_std": float(trend.std(ddof=0)),
                "seasonality_strength": seasonality_strength,
            }
        )

    out = pd.DataFrame(rows).sort_values(["product", "area"]).reset_index(drop=True)
    return out
