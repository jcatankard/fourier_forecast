from fourier_forecast.fourier_forecast import FourierForecast
from .create_data import create_data
import numpy as np
import unittest


class TestFourierForecast(unittest.TestCase):
    atol = 0.1
    rtol = 0.1
    n_tests = 10

    @staticmethod
    def get_fourier_terms() -> dict[str, int]:
        return {
            'weekly_seasonality_terms': np.random.choice(3),
            'monthly_seasonality_terms': np.random.choice(3),
            'quarterly_seasonality_terms': np.random.choice(5),
            'yearly_seasonality_terms': np.random.choice(10)
        }

    def test_basic(self):
        for i in range(self.n_tests):
            print(f'basic tests: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            _, y, _ = create_data(regressors=False, fourier_terms=list(fourier_terms.values()))
            ff = FourierForecast(**fourier_terms)
            ff.fit(y)

            assert np.abs(ff.fitted() - y).mean() < self.atol
            np.testing.assert_allclose(ff.fitted(), y, atol=self.atol, rtol=self.rtol)

    def test_regressors(self):
        for i in range(self.n_tests):
            print(f'regressors tests: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            _, y, r = create_data(regressors=True, fourier_terms=list(fourier_terms.values()))
            ff = FourierForecast(**fourier_terms)
            ff.fit(y, regressors=r)

            assert np.abs(ff.fitted() - y).mean() < self.atol
            np.testing.assert_allclose(ff.fitted(), y, atol=self.atol, rtol=self.rtol)

    def test_sample_weight(self):
        for i in range(self.n_tests):
            print(f'sample weight test: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            _, y, _ = create_data(regressors=False, fourier_terms=list(fourier_terms.values()))

            # if apply zero weight to the first n values, then we can multiply them by a random number and it
            # should not impact the prediction against the last (y.size - n) values
            n = np.random.randint(y.size // 4, y.size - 366)
            w = np.concatenate([np.zeros(n), np.ones(y.size - n)], axis=0)
            y[: n] *= np.random.normal(1., 1., n)

            ff = FourierForecast(**fourier_terms)
            ff.fit(y, sample_weight=w)

            assert np.abs(ff.fitted() - y).mean() < self.atol
            np.testing.assert_allclose(ff.fitted()[n:], y[n:], atol=self.atol, rtol=self.rtol)

    def test_log_y(self):
        for i in range(self.n_tests):
            print(f'log y tests: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            _, y, _ = create_data(regressors=False, fourier_terms=list(fourier_terms.values()), log_y=True)
            ff = FourierForecast(**fourier_terms, log_y=True)
            ff.fit(y)

            assert np.abs(ff.fitted() - y).mean() < self.atol
            np.testing.assert_allclose(ff.fitted(), y, atol=self.atol, rtol=self.rtol)

    def test_regularization(self):
        for i in range(self.n_tests):
            print(f'regularization tests: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            _, y, _ = create_data(regressors=False, fourier_terms=list(fourier_terms.values()))
            y += np.random.normal(0, 10, y.size)

            ff1 = FourierForecast(seasonality_reg=0,
                                  regressor_reg=0,
                                  trend_reg=100,
                                  **fourier_terms
                                  )
            ff1.fit(y)

            ff2 = FourierForecast(**fourier_terms)
            ff2.fit(y)

            assert np.absolute(ff1.params_[1]) < np.absolute(ff2.params_[1])

    def test_trend(self):
        for i in range(self.n_tests):
            print(f'trend tests: {i + 1}')
            fourier_terms = self.get_fourier_terms()
            trend_type = np.random.choice(['linear', 'flat', 'logarithmic', 'logistic'])
            _, y, _ = create_data(regressors=False, fourier_terms=list(fourier_terms.values()), growth=trend_type)
            ff = FourierForecast(growth=trend_type, **fourier_terms)
            ff.fit(y)

            assert np.abs(ff.fitted() - y).mean() < self.atol
            np.testing.assert_allclose(ff.fitted(), y, atol=self.atol, rtol=self.rtol)

    def test_lags(self):
        for i in range(self.n_tests):
            print(f'lag tests: {i + 1}')
            n_lags = np.random.randint(1, 10)

            _, y, r = create_data(regressors=True, fourier_terms=[0, 0, 0, 0], n_lags=n_lags)

            ff = FourierForecast(n_lags=n_lags)
            ff.fit(y, regressors=r)

            fitted = ff.fitted()[n_lags:]
            assert np.abs(fitted - y[n_lags:]).mean() < self.atol
            np.testing.assert_allclose(fitted, y[n_lags:], atol=self.atol, rtol=self.rtol)
