from fourier_forecast.fourier_forecast import FourierForecast, sin_wave
from datetime import date, timedelta
from numpy.typing import NDArray
import numpy as np
import unittest


def make_wave(t: NDArray[np.int64], freq: float) -> NDArray[np.float64]:
    a = np.random.rand() * 10
    p = np.random.normal(0, 2)
    return sin_wave(a, p, freq, t)


def make_regressors(t: NDArray[np.int64], regressors: bool) -> NDArray:
    p = np.random.rand() if regressors else 1.
    n_features = np.random.randint(1, 6) if regressors else 0
    return np.random.choice([0, 1], size=t.size * n_features, p=[p, 1 - p]).reshape(t.size, -1)


def create_data(regressors: bool) -> tuple[NDArray[date], NDArray[np.float64], NDArray[np.float64]]:

    size = np.random.randint(366 * 2, 366 * 3)
    ds = np.arange(date.today() - timedelta(days=size), date.today()).astype('O')
    t = np.arange(size, dtype=np.int64)
    regressor_x = make_regressors(t, regressors)

    y_clean = np.array([
        make_wave(t, 1 / 7),  # weeks
        make_wave(t, 1 / 30.43),  # months
        make_wave(t, 1 / 91.31),  # quarters
        make_wave(t, 1 / 365.25),  # years
        np.random.normal(0, 1) * t,  # trend
        np.random.choice([-1, 1]) * np.random.randint(1, 100) * np.ones(size),  # bias
        np.sum(regressor_x, axis=1)  # regressors
    ]).sum(axis=0)

    return ds, y_clean, regressor_x


class TestFourierForecast(unittest.TestCase):

    atol = 1e-01
    rtol = 1e-01
    n_tests = 3

    def run_test(self, name: str, regressors: bool):
        for i in range(self.n_tests):
            print(f'{name}: {i + 1}')
            ds, y, r = create_data(regressors)
            ff = FourierForecast(monthly_seasonality=True, quarterly_seasonality=True)
            ff.fit(ds, y, regressors=r if regressors else None)
            preds = ff.predict(ds, regressors=r if regressors else None)
            np.testing.assert_allclose(y, preds, atol=self.atol, rtol=self.rtol)

    def test_basic(self):
        self.run_test(name='basic test', regressors=False)

    def test_regressors(self):
        self.run_test(name='regressors test', regressors=True)
