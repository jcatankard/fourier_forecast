from fourier_forecast.predict import predict, create_trend, create_bias
from fourier_forecast.gradient_descent import gradient_descent
from fourier_forecast.waves import sin_wave
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np


class FourierForecast:

    def __init__(self,
                 weekly_seasonality_terms: int = 3,
                 monthly_seasonality_terms: int = 0,
                 quarterly_seasonality_terms: int = 0,
                 yearly_seasonality_terms: int = 10,
                 learning_rate: float = 0.001,
                 n_iterations: int = 100_000,
                 tol: float = 1e-05
                 ):
        self.learning_rate: float = float(learning_rate)
        self.n_iterations: int = int(n_iterations)
        self.tol = float(tol)

        seasonality_terms = {
            7.: weekly_seasonality_terms,
            30.43: monthly_seasonality_terms,
            91.31: quarterly_seasonality_terms,
            365.25: yearly_seasonality_terms
        }
        self.seasonality_terms = {k: v for k, v in seasonality_terms.items() if v > 0}
        self.amplitudes: Optional[NDArray[np.float64]] = None
        self.phases: Optional[NDArray[np.float64]] = None
        self.frequencies: Optional[NDArray[np.float64]] = None
        self.trend: Optional[float] = None
        self.bias: Optional[float] = None

        self.regressors: Optional[NDArray[np.float64]] = None
        self.regressor_weights: Optional[NDArray[np.float64]] = None
        self.n_regressors: Optional[int] = None

        self.ds: Optional[NDArray[np.int64]] = None
        self.pred_start: Optional[int] = None
        self.y: Optional[NDArray[np.float64]] = None
        self.sample_weight: Optional[NDArray[np.float64]] = None

    @staticmethod
    def _to_numpy(a) -> NDArray[np.float64]:
        return np.asarray(a, dtype=np.float64, order='C')

    def _initiate_seasonality_estimates(self):
        n_waves = 2 * sum(self.seasonality_terms.values())
        self.amplitudes = np.zeros(n_waves, dtype=np.float64)
        self.phases = np.zeros(n_waves, dtype=np.float64)
        self.frequencies = np.zeros(n_waves, dtype=np.float64)
        y_detrend = self.y - (self.ds * self.trend + self.bias)
        count = 0
        for periods, terms in self.seasonality_terms.items():
            for j in range(terms):
                f = (j + 1) / periods
                self.frequencies[count] = f
                self.frequencies[count + 1] = f
                self.phases[count] = 0
                self.phases[count + 1] = np.pi

                windows = np.lib.stride_tricks.sliding_window_view(y_detrend, int(1 / f))
                amp_int = (windows.max(axis=1) - windows.min(axis=1)).mean() / 2
                self.amplitudes[count] = 1  # amp_int / 2
                self.amplitudes[count + 1] = 1  # amp_int / 2
                count += 2

    def _initiate_trend_estimates(self):
        x = np.concatenate((np.ones((self.y.size, 1), dtype=np.float64),
                            np.arange(self.y.size, dtype=np.float64).reshape(-1, 1)
                            ), axis=1
                           )
        self.bias, self.trend = np.linalg.inv(x.T @ x) @ x.T @ self.y

    def _initiate_regressors(self, regressors: Optional[NDArray], size: int, default_width: int) -> NDArray[np.float64]:
        regressors = np.zeros((size, default_width), dtype=np.float64) \
            if regressors is None else self._to_numpy(regressors)

        if self.n_regressors is None:
            self.n_regressors = regressors.shape[1]
            self.regressor_weights = np.zeros(self.n_regressors, dtype=np.float64)

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

    def fit(self,
            y: NDArray[np.float64],
            regressors: Optional[NDArray[np.float64]] = None,
            sample_weight: Optional[NDArray[np.float64]] = None
            ):
        self.y = self._to_numpy(y)
        self._initiate_sample_weight(sample_weight)

        self.pred_start = y.size
        self.ds = np.arange(0, y.size, dtype=np.int64)

        self._initiate_trend_estimates()
        self._initiate_seasonality_estimates()
        self.regressors = self._initiate_regressors(regressors, y.size, 0)

        results = gradient_descent(self.bias,
                                   self.trend,
                                   self.amplitudes,
                                   self.phases,
                                   self.frequencies,
                                   self.regressors,
                                   self.regressor_weights,
                                   self.y,
                                   self.learning_rate,
                                   self.n_iterations,
                                   self.tol,
                                   self.sample_weight
                                   )
        self.bias, self.trend, self.amplitudes, self.phases, self.frequencies, self.regressor_weights = results

    def fitted(self) -> NDArray[np.float64]:
        return predict(0,
                       self.y.size,
                       self.bias,
                       self.trend,
                       self.amplitudes,
                       self.phases,
                       self.frequencies,
                       self.regressors,
                       self.regressor_weights
                       )

    def predict(self, h: int = 1, regressors: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        regressors = self._initiate_regressors(regressors, h, self.n_regressors)
        return predict(self.pred_start,
                       h,
                       self.bias,
                       self.trend,
                       self.amplitudes,
                       self.phases,
                       self.frequencies,
                       regressors,
                       self.regressor_weights
                       )

    @staticmethod
    def subplot(ax, ds: NDArray[np.int64], data: NDArray[np.float64], label: str):
        ax.plot(ds, data, label=label)
        ax.legend()

    def plot_components(self):
        n_seasonalities = len(self.seasonality_terms)
        n_rows = 2 + np.ceil(n_seasonalities / 2).astype(np.int64)
        fig, ax = plt.subplots(nrows=n_rows, ncols=2)

        self.subplot(ax[0, 0], self.ds, create_bias(self.bias, self.ds), 'bias')
        self.subplot(ax[0, 1], self.ds, create_trend(self.trend, self.ds), 'trend')
        self.subplot(ax[1, 0], self.ds, self.regressors @ self.regressor_weights, 'regressors')
        self.subplot(ax[1, 1], self.ds, self.y - self.fitted(), 'noise')

        n = 0
        for i, (periods, terms) in enumerate(self.seasonality_terms.items()):
            n_to_plot = np.ceil(periods).astype(np.int64)
            ds = np.arange(n_to_plot)
            s = np.zeros_like(ds, np.float64)
            for j in range(terms):
                s += sin_wave(self.amplitudes[n], self.phases[n], self.frequencies[n], ds)
                s += sin_wave(self.amplitudes[n + 1], self.phases[n + 1], self.frequencies[n + 1], ds)
                n += 2
            row = 2 + i // 2
            col = i % 2
            self.subplot(ax[row, col], ds, s, f'seasonality: periods={periods}')

        if n_seasonalities % 2 == 1:
            ax[n_rows - 1, 1].axis('off')
        plt.show()
