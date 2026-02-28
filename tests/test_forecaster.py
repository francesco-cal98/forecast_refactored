import numpy as np
import pandas as pd
from backend.model.forecaster import Forecaster

def test_forecaster_predict_shape():
    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    f = Forecaster()
    f.model.fit(X, y)
    y_pred, _ = f.predict(X)
    assert y_pred.shape == y.shape


def test_fit_passes_sample_weight_when_supported():
    class SupportsSampleWeightModel:
        def __init__(self):
            self.sample_weight_seen = None

        def fit(self, X, y, sample_weight=None):
            self.sample_weight_seen = sample_weight
            return self

    f = Forecaster()
    f.model = SupportsSampleWeightModel()
    X = pd.DataFrame(np.random.rand(6, 2), columns=["a", "b"])
    y = pd.Series(np.random.rand(6))
    w = np.linspace(0.5, 1.0, 6)

    f.fit(X, y, sample_weight=w)

    assert np.array_equal(f.model.sample_weight_seen, w)


def test_fit_falls_back_when_sample_weight_not_supported():
    class NoSampleWeightModel:
        def __init__(self):
            self.fit_called = False

        def fit(self, X, y):
            self.fit_called = True
            return self

    f = Forecaster()
    f.model = NoSampleWeightModel()
    X = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
    y = pd.Series(np.random.rand(5))
    w = np.linspace(0.5, 1.0, 5)

    f.fit(X, y, sample_weight=w)

    assert f.model.fit_called is True
