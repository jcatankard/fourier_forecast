from fourier_forecast.plot import plot_components, plot_regressor_components, plot_seasonality_components
from numpy.typing import NDArray
import plotly.graph_objs as go
from typing import Optional
import numpy as np
import warnings


DAYS_IN_WEEK = 7.
DAYS_IN_MONTH = 30.43
DAYS_IN_QUARTER = 91.31
DAYS_IN_YEAR = 365.25


class FourierForecast:

    def __init__(self,
                 weekly_seasonality_terms: int = 3,
                 monthly_seasonality_terms: int = 0,
                 quarterly_seasonality_terms: int = 0,
                 yearly_seasonality_terms: int = 10,
                 log_y: bool = False
                 ):

        seasonality_terms = {
            DAYS_IN_WEEK: weekly_seasonality_terms,
            DAYS_IN_MONTH: monthly_seasonality_terms,
            DAYS_IN_QUARTER: quarterly_seasonality_terms,
            DAYS_IN_YEAR: yearly_seasonality_terms
        }
        self.seasonality_terms = {k: v for k, v in seasonality_terms.items() if v > 0}
        self.n_waves = 2 * sum(self.seasonality_terms.values())

        self.log_y = log_y

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

        if (y.min() <= 0) and self.log_y:
            self.log_y = False
            warnings.warn(f'For log_y=True, values of y must be > 0. Found value: {y.min()}. Setting log_y to False.')

        self.y = self._to_numpy(np.log(y) if self.log_y else y)
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
        y_hat = self.x_ @ self.params_
        return np.exp(y_hat) if self.log_y else y_hat

    def predict(self, h: int = 1, regressors: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:

        ds = np.arange(self.pred_start, self.pred_start + h, dtype=np.int64)

        x = np.concatenate([
            self._initiate_bias(ds),
            self._initiate_trend(ds),
            self._initiate_seasonalities(ds),
            self._initiate_regressors(regressors, h)
        ], axis=1)

        preds = x @ self.params_
        return np.exp(preds) if self.log_y else preds

    def plot_components(self) -> go.Figure:
        return plot_components(self)

    def plot_regressor_components(self, regressor_names: Optional[list[str]] = None) -> go.Figure:
        return plot_regressor_components(self, regressor_names)

    def plot_seasonality_components(self) -> go.Figure:
        return plot_seasonality_components(self)
