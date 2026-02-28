from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .io import load_area_panel
from .plots import (
    plot_price_sales_facets,
    plot_promo_impact_facets,
    plot_seasonality_facets,
    plot_trend_facets,
)
from .quality import run_quality_checks
from .summary import build_series_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run per-series EDA for praenova_validation_2y.")
    parser.add_argument(
        "--input",
        type=str,
        default="praenova_validation_2y.xlsx",
        help="Path to the source workbook.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eda/praenova_validation_2y",
        help="Output folder for EDA artifacts.",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2023,
        help="Base year used to convert Year/Week into ISO dates.",
    )
    return parser


def run_eda(input_path: str, output_dir: str, base_year: int = 2023) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_area_panel(path=input_path, base_year=base_year)
    quality_report = run_quality_checks(df)
    series_summary = build_series_summary(df)

    summary_fp = out_dir / "series_summary.csv"
    quality_fp = out_dir / "quality_report.json"
    trend_fp = out_dir / "dashboard_trend.png"
    seasonality_fp = out_dir / "dashboard_seasonality.png"
    promo_fp = out_dir / "dashboard_promo_impact.png"
    price_fp = out_dir / "dashboard_price_vs_sales.png"

    series_summary.to_csv(summary_fp, index=False)
    with quality_fp.open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, sort_keys=True)

    fig_trend = plot_trend_facets(df)
    fig_trend.savefig(trend_fp, bbox_inches="tight", dpi=150)
    plt.close(fig_trend)

    fig_seasonality = plot_seasonality_facets(df)
    fig_seasonality.savefig(seasonality_fp, bbox_inches="tight", dpi=150)
    plt.close(fig_seasonality)

    fig_promo = plot_promo_impact_facets(df)
    fig_promo.savefig(promo_fp, bbox_inches="tight", dpi=150)
    plt.close(fig_promo)

    fig_price = plot_price_sales_facets(df)
    fig_price.savefig(price_fp, bbox_inches="tight", dpi=150)
    plt.close(fig_price)

    return {
        "series_summary.csv": str(summary_fp),
        "quality_report.json": str(quality_fp),
        "dashboard_trend.png": str(trend_fp),
        "dashboard_seasonality.png": str(seasonality_fp),
        "dashboard_promo_impact.png": str(promo_fp),
        "dashboard_price_vs_sales.png": str(price_fp),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = run_eda(input_path=args.input, output_dir=args.output, base_year=args.base_year)
    for name, fp in paths.items():
        print(f"{name}: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
