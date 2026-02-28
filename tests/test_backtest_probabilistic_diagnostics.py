import numpy as np

from backend.backtest import BacktestRunner


def test_aggregate_probabilistic_diagnostics_with_intervals():
    y_true_cuts = [
        np.array([10.0, 20.0, 30.0]),
        np.array([12.0, 18.0]),
    ]
    p50_cuts = [
        np.array([11.0, 19.0, 29.0]),
        np.array([10.0, 20.0]),
    ]
    p10_cuts = [
        np.array([8.0, 18.0, 28.0]),
        np.array([9.0, 19.0]),
    ]
    p90_cuts = [
        np.array([12.0, 22.0, 32.0]),
        np.array([11.0, 21.0]),
    ]

    out = BacktestRunner._aggregate_probabilistic_diagnostics(
        y_true_cuts=y_true_cuts,
        p50_cuts=p50_cuts,
        p10_cuts=p10_cuts,
        p90_cuts=p90_cuts,
    )

    assert np.allclose(out["abs_error_by_h"], [1.5, 1.5, 1.0])
    assert np.allclose(out["coverage_by_h"], [0.5, 0.5, 1.0])
    assert np.isclose(out["coverage_10_90"], 0.6)
    assert out["coverage_available"] is True


def test_aggregate_probabilistic_diagnostics_without_intervals():
    out = BacktestRunner._aggregate_probabilistic_diagnostics(
        y_true_cuts=[np.array([1.0, 2.0])],
        p50_cuts=[np.array([1.5, 1.0])],
        p10_cuts=[None],
        p90_cuts=[None],
    )

    assert np.allclose(out["abs_error_by_h"], [0.5, 1.0])
    assert out["coverage_by_h"] is None
    assert np.isnan(out["coverage_10_90"])
    assert out["coverage_available"] is False
