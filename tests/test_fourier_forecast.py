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

    atol = 0.1
    rtol = 0.1
    n_tests = 3

    def test_basic(self):
        for i in range(self.n_tests):
            print(f'basic tests: {i + 1}')
            ds, y, _ = create_data(regressors=False)
            ff = FourierForecast(monthly_seasonality=True, quarterly_seasonality=True)
            ff.fit(ds, y)
            preds = ff.predict(ds)
            np.testing.assert_allclose(y, preds, atol=self.atol, rtol=self.rtol)

    def test_regressors(self):
        for i in range(self.n_tests):
            print(f'regressors tests: {i + 1}')
            ds, y, r = create_data(regressors=True)
            ff = FourierForecast(monthly_seasonality=True, quarterly_seasonality=True)
            ff.fit(ds, y, regressors=r)
            preds = ff.predict(ds, regressors=r)
            np.testing.assert_allclose(y, preds, atol=self.atol, rtol=self.rtol)

    def test_sample_weight(self):
        for i in range(self.n_tests):
            print(f'sample weight test: {i + 1}')
            ds, y, _ = create_data(regressors=False)

            # if apply zero weight to the first x values, then we can multiply them by a random number and it
            # should not impact the prediction against the last (y.size - x) values
            x = np.random.randint(y.size // 4, y.size - 366)
            w = np.concatenate([np.zeros(x), np.ones(y.size - x)], axis=0)
            y[: x] *= np.random.rand()

            ff = FourierForecast(monthly_seasonality=True, quarterly_seasonality=True)
            ff.fit(ds, y, sample_weight=w)

            preds = ff.predict(ds)
            np.testing.assert_allclose(y[x:], preds[x:], atol=self.atol, rtol=self.rtol)
