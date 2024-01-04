from fourier_forecast.fourier_forecast import FourierForecast
from create_data import create_data
import numpy as np
import unittest


class TestFourierForecast(unittest.TestCase):

    atol = 0.1
    rtol = np.inf
    n_tests = 10

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

            # if apply zero weight to the first n values, then we can multiply them by a random number and it
            # should not impact the prediction against the last (y.size - n) values
            n = np.random.randint(y.size // 4, y.size - 366)
            w = np.concatenate([np.zeros(n), np.ones(y.size - n)], axis=0)
            y[: n] *= np.random.normal(1., 1., n)

            ff = FourierForecast(*fourier_terms)
            ff.fit(y, sample_weight=w)
            np.testing.assert_allclose(y[n:], ff.fitted()[n:], atol=self.atol, rtol=self.rtol)
