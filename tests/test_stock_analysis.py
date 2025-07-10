import numpy as np
import pandas as pd
from backend.model.forecaster import Forecaster

def test_stock_analysis_logic():
    y_test = pd.Series([100, 120, 110, 130, 90])
    y_pred_test = pd.Series([95, 125, 105, 135, 85])
    y_train = pd.Series([80, 90, 100, 110, 95])
    y_pred_train = pd.Series([85, 88, 98, 108, 92])
    sell_in_true = y_test + 10

    f = Forecaster()
    f.model.fit(np.random.rand(5, 2), y_train)  # fake fit

    result = f.stock_analysis(
        y_test, y_pred_test, y_train, y_pred_train,
        sell_in_true,
        safety_stock_time=2,
        lead_time=1,
        expected_service_level=0.9
    )

    assert isinstance(result, dict)
    assert "lost_sales_pct" in result
    assert 0 <= result["lost_sales_pct"] <= 100
    assert all(len(result[k]) == len(y_test) for k in ["safety_stock", "forecasted_stock", "actual", "predicted"])
