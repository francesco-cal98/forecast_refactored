import numpy as np
from backend.utils.metrics import MAPE, WAPE

def test_mape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = MAPE(y_true, y_pred)
    assert 0 <= result <= 100

def test_wape():
    y_true = np.array([50, 100, 150])
    y_pred = np.array([48, 102, 149])
    result = WAPE(y_true, y_pred)
    assert 0 <= result <= 100
