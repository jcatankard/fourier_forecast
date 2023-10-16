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
                 multiplicative_seasonality: bool = False,
                 learning_rate: float = 0.001,
                 n_iterations: int = 100_000
                 ):
        self.multiplicative_seasonality: bool = multiplicative_seasonality
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations

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

    def fit(self, ds: list[date] | NDArray[date], y: NDArray[np.float64], regressors: NDArray[np.float64] = None):
        y = self._to_numpy(y)
        self.y = np.log(y) if self.multiplicative_seasonality else y

        self.min_date = min(ds)
        self.ds = np.array([(d - self.min_date).days for d in ds], dtype=np.int64)

        self._initiate_seasonality_estimates()
        self._initiate_trend_estimates()

        regressors = np.zeros((y.size, 0), dtype=np.float64) \
            if regressors is None else self._to_numpy(regressors)
        self.regressors = np.log(regressors) if self.multiplicative_seasonality else regressors
        self.n_regressors = self.regressors.shape[1]
        self.regressor_weights = np.zeros(self.n_regressors, dtype=np.float64)

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
                                   self.n_iterations
                                   )
        self.bias, self.trend, self.amplitudes, self.phases, self.frequencies, self.regressor_weights = results

    def predict(self, ds: list[date] | NDArray[date], regressors: NDArray[np.float64] = None
                ) -> NDArray[np.float64]:
        ds = np.array([(d - self.min_date).days for d in ds], dtype=np.int64)
        regressors = np.zeros(shape=(ds.size, self.n_regressors), dtype=np.float64) \
            if regressors is None else regressors
        regressors = np.log(regressors) if self.multiplicative_seasonality else regressors

        preds = predict(ds,
                        self.bias,
                        self.trend,
                        self.amplitudes,
                        self.phases,
                        self.frequencies,
                        regressors,
                        self.regressor_weights
                        )
        return np.exp(preds) if self.multiplicative_seasonality else preds

    def plot_components(self):
        ds = np.array([self.min_date + timedelta(days=int(d)) for d in self.ds])
        fig, ax = plt.subplots(nrows=3, ncols=2)

        b = create_bias(self.bias, self.ds)
        b = np.exp(b) if self.multiplicative_seasonality else b
        ax[0, 0].plot(ds, b, label='bias')
        ax[0, 0].set_title('bias')

        t = create_trend(self.trend, self.ds)
        t = np.exp(t) if self.multiplicative_seasonality else t
        ax[0, 1].plot(ds, t, label='trend')
        ax[0, 1].set_title('trend')

        for w in range(self.amplitudes.size):
            s = sin_wave(self.amplitudes[w], self.phases[w], self.frequencies[w], self.ds)
            s = np.exp(s) if self.multiplicative_seasonality else s
            ax[1, 0].plot(ds, s, label=f'seasonality {w + 1}')
        ax[1, 0].set_title('seasonality')

        r = self.regressors @ self.regressor_weights
        r = np.exp(r) if self.multiplicative_seasonality else r
        ax[1, 1].plot(ds, r, label='regressors')
        ax[1, 1].set_title('regressors')

        regressors = np.exp(self.regressors) if self.multiplicative_seasonality else self.regressors
        y = np.exp(self.y) if self.multiplicative_seasonality else self.y
        ax[2, 0].plot(ds, y - self.predict(ds, regressors), label='noise')
        ax[2, 0].set_title('noise')

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
