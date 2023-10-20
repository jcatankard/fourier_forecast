from fourier_forecast.predict import predict, create_trend, create_bias
from fourier_forecast.gradient_descent import gradient_descent
from fourier_forecast.waves import sin_wave
from datetime import date, timedelta
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


class FourierForecast:

    def __init__(self,
                 weekly_seasonality: bool = True,
                 monthly_seasonality: bool = False,
                 quarterly_seasonality: bool = False,
                 yearly_seasonality: bool = True,
                 learning_rate: float = 0.001,
                 n_iterations: int = 100_000,
                 tol: float = 1e-05
                 ):
        self.learning_rate: float = float(learning_rate)
        self.n_iterations: int = int(n_iterations)
        self.tol = float(tol)

        self.time_periods = np.array([7, 30.43, 91.31, 365.25]
                                     )[[weekly_seasonality,
                                        monthly_seasonality,
                                        quarterly_seasonality,
                                        yearly_seasonality
                                        ]]
        self.amplitudes: NDArray[np.float64] = None
        self.phases: NDArray[np.float64] = None
        self.frequencies = 1 / self.time_periods
        self.trend: float = None
        self.bias: float = None

        self.regressors: NDArray[np.float64] = None
        self.regressor_weights: NDArray[np.float64] = None
        self.n_regressors: int = None

        self.ds: NDArray[np.int64] = None
        self.min_date: date = None
        self.y: NDArray[np.float64] = None
        self.sample_weight: NDArray[np.float64] = None
        self.trend_estimate: NDArray[np.float64] = None

    @staticmethod
    def _to_numpy(a) -> NDArray[np.float64]:
        return np.asarray(a, dtype=np.float64, order='C')

    def _initiate_seasonality_estimates(self):
        self.amplitudes = np.zeros(self.time_periods.size, dtype=np.float64)
        self.phases = np.zeros(self.time_periods.size, dtype=np.float64)
        self.trend_estimate = self.y.copy()
        for i, t in enumerate(self.time_periods):
            if self.y.size > np.ceil(t):
                size_options = np.arange(np.floor(t).astype(np.int64), self.y.size + 1)
                div = size_options / t
                diff = np.absolute(div - div.round(0))
                size = size_options[diff == diff.min()].max()
                a, p, f = find_signal(self.trend_estimate[: size], t)
                self.amplitudes[i] = a
                self.phases[i] = p
                self.trend_estimate -= sin_wave(a, p, self.frequencies[i], self.ds)

    def _initiate_trend_estimates(self):
        gradient = (self.trend_estimate[-1] - self.trend_estimate[0]) / self.trend_estimate.size
        self.trend = gradient
        self.bias = (self.trend_estimate - gradient * self.ds).mean()

    def _initiate_regressors(self, regressors: NDArray, size: int, default_width: int) -> NDArray[np.float64]:
        regressors = np.zeros((size, default_width), dtype=np.float64) \
            if regressors is None else self._to_numpy(regressors)

        if self.n_regressors is None:
            self.n_regressors = regressors.shape[1]
            self.regressor_weights = np.zeros(self.n_regressors, dtype=np.float64)

        return regressors

    def _initiate_sample_weight(self, sample_weight: NDArray[np.float64]):
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
            ds: list[date] | NDArray[date],
            y: NDArray[np.float64],
            regressors: NDArray[np.float64] = None,
            sample_weight: NDArray[np.float64] = None
            ):
        self.y = self._to_numpy(y)
        self._initiate_sample_weight(sample_weight)

        self.min_date = min(ds)
        self.ds = np.array([(d - self.min_date).days for d in ds], dtype=np.int64)

        self._initiate_seasonality_estimates()
        self._initiate_trend_estimates()
        self.regressors = self._initiate_regressors(regressors, y.size, 0)

        results = gradient_descent(self.ds,
                                   self.bias,
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

    def predict(self, ds: list[date] | NDArray[date], regressors: NDArray[np.float64] = None
                ) -> NDArray[np.float64]:
        ds = np.array([(d - self.min_date).days for d in ds], dtype=np.int64)
        regressors = self._initiate_regressors(regressors, ds.size, self.n_regressors)
        return predict(ds,
                       self.bias,
                       self.trend,
                       self.amplitudes,
                       self.phases,
                       self.frequencies,
                       regressors,
                       self.regressor_weights
                       )

    @staticmethod
    def subplot(ax, ds: NDArray, data: NDArray, label: str, title: str = None):
        ax.plot(ds, data, label=label)
        ax.set_title(label if title is None else title)

    def plot_components(self):
        ds = np.array([self.min_date + timedelta(days=int(d)) for d in self.ds])
        fig, ax = plt.subplots(nrows=3, ncols=2)

        self.subplot(ax[0, 0], ds, create_bias(self.bias, self.ds), 'bias')
        self.subplot(ax[0, 1], ds, create_trend(self.trend, self.ds), 'trend')

        for w in range(self.amplitudes.size):
            s = sin_wave(self.amplitudes[w], self.phases[w], self.frequencies[w], self.ds)
            self.subplot(ax[1, 0], ds, s, f'seasonality {w + 1}', 'seasonality')

        self.subplot(ax[1, 1], ds, self.regressors @ self.regressor_weights, 'regressors')
        self.subplot(ax[2, 0], ds, self.y - self.predict(ds, self.regressors), 'noise')
        ax[2, 1].axis('off')
        plt.show()

    def print(self):
        print('--------------------')
        print('periods:', 1 / self.frequencies)
        print('amplitudes:', self.amplitudes.round(3))
        print('phases:', self.phases.round(3))
        print('trend:', np.round(self.trend, 5))
        print('bias:', np.round(self.bias, 5))
        print('--------------------')


def find_signal(a: NDArray, periods: float):
    """returns the amplitude, phase and freq of a wave for a given (non-integer) no. periods"""
    yhat = np.fft.rfft(a)
    # power_spectral_density = np.abs(yhat) / a.size

    frequencies = np.fft.rfftfreq(a.size, d=1 / periods)
    amplitudes = np.abs(yhat) * 2 / a.size
    phases = np.arctan2(yhat.imag, yhat.real) + np.pi / 2

    f = np.absolute(frequencies - 1)
    i = np.where(f == f.min())[0][0]
    return amplitudes[i], phases[i], frequencies[i] / periods
