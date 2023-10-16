from fourier_forecast.waves import sin_wave
from numba import njit, int64, float64
from numpy.typing import NDArray
import numpy as np


@njit(float64[::1](float64, int64[::1]), cache=True)
def create_bias(bias: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    return np.ones_like(ds, dtype=np.float64) * bias


@njit(float64[::1](float64, int64[::1]), cache=True)
def create_trend(trend: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    return trend * ds


@njit(float64[::1](
    int64[::1],
    float64,
    float64,
    float64[::1],
    float64[::1],
    float64[::1],
    float64[:, ::1],
    float64[::1]
), cache=True)
def predict(ds: NDArray[np.int64],
            bias: float,
            trend: float,
            amplitudes: NDArray[np.float64],
            phases: NDArray[np.float64],
            frequencies: NDArray[np.float64],
            regressors: NDArray[np.float64],
            regressor_weights: NDArray[np.float64]
            ) -> NDArray[np.float64]:
    # add bias
    y_pred = create_bias(bias, ds)
    # add trend
    y_pred += create_trend(trend, ds)
    # add seasonality
    for w in range(amplitudes.size):
        y_pred += sin_wave(amplitudes[w], phases[w], frequencies[w], ds)
    # add regressors
    y_pred += regressors @ regressor_weights
    return y_pred
