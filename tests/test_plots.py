import matplotlib
import numpy as np

matplotlib.use("Agg")

from backend.utils.plots import (
    visualize_forecast_fanchart,
    visualize_inventory_scenarios,
    plot_coverage,
    plot_error_vs_horizon,
)


def test_visualize_forecast_fanchart_smoke(tmp_path):
    y_true = np.array([12.0, 15.0, 10.0, 18.0])
    p50 = np.array([11.0, 14.0, 12.0, 16.0])
    p10 = p50 - 3.0
    p90 = p50 + 3.0
    save_path = tmp_path / "fanchart.png"

    fig = visualize_forecast_fanchart(
        product="TEST_SKU",
        y_true=y_true,
        p50=p50,
        p10=p10,
        p90=p90,
        save_path=str(save_path),
    )

    assert fig is not None
    assert save_path.exists()


def test_plot_coverage_and_error_vs_horizon_smoke(tmp_path):
    fig_cov = plot_coverage(
        product="TEST_SKU",
        coverage_global=0.78,
        expected=0.80,
        coverage_by_h=np.array([0.7, 0.8, 0.85]),
    )
    fig_err = plot_error_vs_horizon(
        product="TEST_SKU",
        abs_error_by_h=np.array([12.0, 16.0, 20.0]),
    )

    cov_path = tmp_path / "coverage.png"
    err_path = tmp_path / "error_h.png"
    fig_cov.savefig(cov_path)
    fig_err.savefig(err_path)

    assert cov_path.exists()
    assert err_path.exists()


def test_visualize_inventory_scenarios_smoke(tmp_path):
    fig = visualize_inventory_scenarios(
        product="TEST_SKU",
        stock_low=np.array([40.0, 32.0, 28.0, 20.0]),
        stock_base=np.array([45.0, 35.0, 25.0, 15.0]),
        stock_high=np.array([50.0, 36.0, 18.0, 0.0]),
        safety_stock_line=np.array([10.0, 10.0, 10.0, 10.0]),
    )
    out = tmp_path / "inventory_band.png"
    fig.savefig(out)
    assert out.exists()
