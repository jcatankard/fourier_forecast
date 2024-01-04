from numpy.typing import NDArray
import matplotlib.pyplot as plt
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

        self.pred_start = y.size
        self.ds = np.arange(0, y.size, dtype=np.int64)

        self.x_ = np.concatenate([
            self._initiate_bias(self.ds),
            self._initiate_trend(self.ds),
            self._initiate_seasonalities(self.ds),
            self._initiate_regressors(regressors, y.size, 0)
        ], axis=1)

        penalty = self._initiate_regularization_penalty()
        self.params_ = np.linalg.inv(self.x_.T @ self.x_ + penalty) @ self.x_.T @ self.y

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

    def _initiate_regressors(self, regressors: Optional[NDArray], size: int, default_width: int) -> NDArray[np.float64]:
        regressors = np.zeros((size, default_width), dtype=np.float64) \
            if regressors is None else self._to_numpy(regressors)

        self.n_regressors = regressors.shape[1]

        return regressors

    def _initiate_sample_weight(self, sample_weight: Optional[NDArray[np.float64]]):
        if sample_weight is None:
            self.sample_weight = np.ones_like(self.y, dtype=np.float64)
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
            self._initiate_regressors(regressors, h, self.n_regressors)
        ], axis=1)

        return x @ self.params_

    @staticmethod
    def subplot(ax, ds: NDArray[np.int64], data: NDArray[np.float64], label: str):
        ax.plot(ds, data, label=label)
        ax.legend()

    def plot_components(self):
        n_seasonalities = len(self.seasonality_terms)
        n_rows = 2 + np.ceil(n_seasonalities / 2).astype(np.int64)
        fig, ax = plt.subplots(nrows=n_rows, ncols=2)

        self.subplot(ax[0, 0], self.ds, self.x_[:, 0] * self.params_[0], 'bias')
        self.subplot(ax[0, 1], self.ds, self.x_[:, 1] * self.params_[1], 'trend')

        regs = self.x_[:, -self.n_regressors:] @ self.params_[-self.n_regressors:]
        self.subplot(ax[1, 0], self.ds, regs, 'regressors')

        self.subplot(ax[1, 1], self.ds, self.y - self.fitted(), 'noise')

        start_col = 2
        for i, (periods, n_terms) in enumerate(self.seasonality_terms.items()):
            n_to_plot = np.ceil(periods).astype(np.int64)
            ds = np.arange(n_to_plot)

            end_col = start_col + n_terms * 2
            s = self.x_[:, start_col: end_col] @ self.params_[start_col: end_col]

            row = 2 + i // 2
            col = i % 2
            self.subplot(ax[row, col], ds, s[: n_to_plot], f'seasonality: periods={periods}')

            start_col = end_col

        if n_seasonalities % 2 == 1:
            ax[n_rows - 1, 1].axis('off')
        plt.show()
