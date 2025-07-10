import numpy as np
from backend.model.forecaster import Forecaster

def test_forecaster_predict_shape():
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    f = Forecaster()
    f.model.fit(X, y)
    y_pred, _ = f.predict(X)
    assert y_pred.shape == y.shape
