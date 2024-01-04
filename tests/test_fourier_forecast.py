from fourier_forecast.fourier_forecast import FourierForecast
from fourier_forecast.waves import sin_wave
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


def create_data(regressors: bool,
                fourier_terms: NDArray
                ) -> tuple[NDArray[date], NDArray[np.float64], NDArray[np.float64]]:

    size = np.random.randint(366 * 2, 366 * 3)
    ds = np.arange(date.today() - timedelta(days=size), date.today()).astype('O')
    t = np.arange(size, dtype=np.int64)
    regressor_x = make_regressors(t, regressors)

    y_clean = np.array([
        np.random.normal(0, .1) * t,  # trend
        np.random.choice([-1, 1]) * np.random.randint(1, 100) * np.ones(size),  # bias
        np.sum(regressor_x, axis=1)  # regressors
    ]).sum(axis=0)

    # weeks, months, quarters, years
    for i, periods in enumerate([7, 30.43, 91.31, 365.25]):
        for f in range(fourier_terms[i]):
            y_clean += make_wave(t, (f + 1) / periods)

    return ds, y_clean, regressor_x


class TestFourierForecast(unittest.TestCase):

    atol = 0.1
    rtol = np.inf
    n_tests = 3

    def test_basic(self):
        for i in range(self.n_tests):
            print(f'basic tests: {i + 1}')
            fourier_terms = np.random.choice(list(range(5)), 4)
            _, y, _ = create_data(regressors=False, fourier_terms=fourier_terms)
            ff = FourierForecast(*fourier_terms)
            ff.fit(y)
            np.testing.assert_allclose(y, ff.fitted(), atol=self.atol, rtol=self.rtol)

    def test_regressors(self):
        for i in range(self.n_tests):
            print(f'regressors tests: {i + 1}')
            fourier_terms = np.random.choice(list(range(3)), 4)
            _, y, r = create_data(regressors=True, fourier_terms=fourier_terms)
            ff = FourierForecast(*fourier_terms)
            ff.fit(y, regressors=r)
            np.testing.assert_allclose(y, ff.fitted(), atol=self.atol, rtol=self.rtol)

    def test_sample_weight(self):
        for i in range(self.n_tests):
            print(f'sample weight test: {i + 1}')
            fourier_terms = np.random.choice(list(range(2)), 4)
            _, y, _ = create_data(regressors=False, fourier_terms=fourier_terms)

            # if apply zero weight to the first x values, then we can multiply them by a random number and it
            # should not impact the prediction against the last (y.size - x) values
            x = np.random.randint(y.size // 4, y.size - 366)
            w = np.concatenate([np.zeros(x), np.ones(y.size - x)], axis=0)
            y[: x] *= np.random.normal(1., 1.)

            ff = FourierForecast(*fourier_terms)
            ff.fit(y, sample_weight=w)
            np.testing.assert_allclose(y[x:], ff.fitted()[x:], atol=self.atol, rtol=self.rtol)
