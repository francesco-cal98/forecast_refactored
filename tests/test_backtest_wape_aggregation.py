import numpy as np

from backend.backtest import BacktestRunner


def test_pooled_wape_differs_from_mean_of_cut_wapes():
    # Cut 1: 10/100 = 10%
    # Cut 2:  3/10  = 30%
    # Mean of cut WAPEs = 20%, pooled WAPE = 100*(10+3)/(100+10) = 11.818...%
    mean_of_cut_wapes = np.mean([10.0, 30.0])
    pooled_wape, valid = BacktestRunner._compute_pooled_wape(abs_error_sum=13.0, abs_actual_sum=110.0)

    assert valid is True
    assert mean_of_cut_wapes != pooled_wape
    assert np.isclose(pooled_wape, 11.818181818181818, rtol=1e-12, atol=0.0)


def test_pooled_wape_handles_zero_actual_sum():
    pooled_wape, valid = BacktestRunner._compute_pooled_wape(abs_error_sum=5.0, abs_actual_sum=0.0)

    assert valid is False
    assert np.isnan(pooled_wape)
