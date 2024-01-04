from fourier_forecast.plot import plot_components
from numpy.typing import NDArray
from typing import Optional
import numpy as np


class FourierForecast:

    def __init__(self,
                 weekly_seasonality_terms: int = 3,
                 monthly_seasonality_terms: int = 0,
                 quarterly_seasonality_terms: int = 0,
                 yearly_seasonality_terms: int = 10
                 ):
        # add params for linear/flat trend, fit-intercept, regularization terms

        seasonality_terms = {
            7.: weekly_seasonality_terms,
            30.43: monthly_seasonality_terms,
            91.31: quarterly_seasonality_terms,
            365.25: yearly_seasonality_terms
        }
        self.seasonality_terms = {k: v for k, v in seasonality_terms.items() if v > 0}
        self.n_waves = 2 * sum(self.seasonality_terms.values())

        self.regressors: Optional[NDArray[np.float64]] = None
        self.n_regressors: Optional[int] = None

        self.ds: Optional[NDArray[np.int64]] = None
        self.pred_start: Optional[int] = None
        self.y: Optional[NDArray[np.float64]] = None
        self.sample_weight: Optional[NDArray[np.float64]] = None
        self.x_: Optional[NDArray[np.float64]] = None
        self.params_: Optional[NDArray[np.float64]] = None

    @staticmethod
    def _to_numpy(a) -> NDArray[np.float64]:
        return np.asarray(a, dtype=np.float64, order='C')

    def fit(self,
            y: NDArray[np.float64],
            regressors: Optional[NDArray[np.float64]] = None,
            sample_weight: Optional[NDArray[np.float64]] = None
            ):
        self.y = self._to_numpy(y)
        self._initiate_sample_weight(sample_weight)
        self.n_regressors = 0 if regressors is None else regressors.shape[1]

        self.pred_start = y.size
        self.ds = np.arange(0, y.size, dtype=np.int64)

        self.x_ = np.concatenate([
            self._initiate_bias(self.ds),
            self._initiate_trend(self.ds),
            self._initiate_seasonalities(self.ds),
            self._initiate_regressors(regressors, y.size)
        ], axis=1)

        self._rescale_data()

        penalty = self._initiate_regularization_penalty()
        self.params_ = np.linalg.inv(self.x_.T @ self.x_ + penalty) @ self.x_.T @ self.y

    def _rescale_data(self):
        """reference: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_base.py"""
        if self.sample_weight is not None:
            sample_weight_sqrt = np.sqrt(self.sample_weight)
            self.x_ *= sample_weight_sqrt[:, np.newaxis]
            self.y *= sample_weight_sqrt[: np.newaxis]

    def _initiate_regularization_penalty(self) -> NDArray[np.int64]:
        penalty = np.identity(self.x_.shape[1])
        penalty[0][0] = 0  # for intercept
        penalty *= 0  # set to zero for now
        return penalty

    def _initiate_seasonalities(self, ds: NDArray[np.int64]) -> NDArray[np.float64]:
        count = 0
        waves = np.empty((self.n_waves, ds.size), dtype=np.float64)
        for periods, terms in self.seasonality_terms.items():
            for j in range(terms):
                f = (j + 1) / periods
                waves[count] = np.sin(2 * np.pi * ds * f, dtype=np.float64)
                waves[count + 1] = np.cos(2 * np.pi * ds * f, dtype=np.float64)
                count += 2

        return waves.T

    @staticmethod
    def _initiate_bias(ds: NDArray[np.int64]) -> NDArray[np.float64]:
        """create intercept. later allow option for no intercept."""
        return np.ones(shape=(ds.size, 1), dtype=np.float64)

    @staticmethod
    def _initiate_trend(ds: NDArray[np.int64]) -> NDArray[np.float64]:
        """create trend array. later allow option for flat trend."""
        return ds.astype(np.float64).reshape(-1, 1)

    def _initiate_regressors(self, regs: Optional[NDArray], size: int) -> NDArray[np.float64]:
        return np.zeros((size, self.n_regressors), dtype=np.float64) if regs is None else self._to_numpy(regs)

    def _initiate_sample_weight(self, sample_weight: Optional[NDArray[np.float64]]):
        if sample_weight is None:
            self.sample_weight = None
        elif min(sample_weight) < 0:
            raise ValueError('All sample weights must be >= 0')
        elif max(sample_weight) == 0:
            raise ValueError('Max sample weight must be greater than zero')
        elif len(sample_weight) != self.y.size:
            raise ValueError('Size of sample weight must be same as time-series data')
        else:
            sample_weight = self._to_numpy(sample_weight)
            self.sample_weight = sample_weight / sample_weight.max()

    def fitted(self) -> NDArray[np.float64]:
        return self.x_ @ self.params_

    def predict(self, h: int = 1, regressors: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:

        ds = np.arange(self.pred_start, self.pred_start + h, dtype=np.int64)

        x = np.concatenate([
            self._initiate_bias(ds),
            self._initiate_trend(ds),
            self._initiate_seasonalities(ds),
            self._initiate_regressors(regressors, h)
        ], axis=1)

        return x @ self.params_

    def plot_components(self):
        plot_components(self)
