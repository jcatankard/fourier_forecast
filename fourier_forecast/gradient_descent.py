from fourier_forecast.waves import dy_da, dy_dp
from numba import njit, int64, float64, types
from fourier_forecast.predict import predict
from numpy.typing import NDArray
import numpy as np


@njit(float64[::1](
    int64[::1],
    float64,
    float64,
    float64[::1],
    float64[::1],
    float64[::1],
    float64[:, ::1],
    float64[::1],
    float64[::1]
), cache=True)
def find_cost_derivative(ds: NDArray[np.int64],
                         bias: float,
                         trend: float,
                         amplitudes: NDArray[np.float64],
                         phases: NDArray[np.float64],
                         frequencies: NDArray[np.float64],
                         regressors: NDArray[np.float64],
                         regressor_weights: NDArray[np.float64],
                         y: NDArray[np.float64]
                         ) -> NDArray[np.float64]:
    """cost function -> mse = (1 / n) * (y_pred - y) ** 2 -> (2/n) * (y_pred - y)"""
    y_pred = predict(ds,
                     bias,
                     trend,
                     amplitudes,
                     phases,
                     frequencies,
                     regressors,
                     regressor_weights
                     )
    return (2 / ds.size) * (y_pred - y)


@njit(float64(float64, float64[::1], int64[::1], float64), cache=True)
def update_trend(trend: float,
                 cost_derivative: NDArray[np.float64],
                 ds: NDArray[np.int64],
                 learning_rate: float
                 ) -> float:
    gradient = ds / ds.size
    d = gradient * cost_derivative
    return trend - d.sum() * learning_rate


@njit(float64[::1](float64[::1], float64[:, ::1], float64[::1], float64), cache=True)
def update_regressor_weights(regressor_weights: NDArray[np.float64],
                             regressors: NDArray[np.float64],
                             cost_derivative: NDArray[np.float64],
                             learning_rate: float
                             ) -> NDArray[np.float64]:
    dw = regressors.T @ cost_derivative
    return regressor_weights - dw * learning_rate


@njit(types.UniTuple(float64[::1], 3)(
    float64[::1],
    float64[::1],
    float64[::1],
    int64[::1],
    float64[::1],
    float64
), cache=True)
def update_waves(amplitudes: NDArray[np.float64],
                 phases: NDArray[np.float64],
                 frequencies: NDArray[np.float64],
                 ds: NDArray[np.int64],
                 cost_derivative: NDArray[np.float64],
                 learning_rate: float
                 ):
    for w in range(amplitudes.size):
        da = dy_da(phases[w], frequencies[w], ds) * cost_derivative
        dp = dy_dp(amplitudes[w], phases[w], frequencies[w], ds) * cost_derivative
        # df = dy_df(amplitudes[w], phases[w], frequencies[w], ds) * cost_derivative

        amplitudes[w] -= da.sum() * learning_rate
        phases[w] -= dp.sum() * learning_rate
        # frequencies[w] -= df.sum() * learning_rate

    return amplitudes, phases, frequencies


@njit(float64(float64, float64[::1], float64), cache=True)
def update_bias(bias: float, cost_derivative: NDArray[np.float64], learning_rate: float) -> float:
    return bias - cost_derivative.sum() * learning_rate


@njit(types.Tuple((float64, float64, float64[::1], float64[::1], float64[::1], float64[::1]))(
    int64[::1],
    float64,
    float64,
    float64[::1],
    float64[::1],
    float64[::1],
    float64[:, ::1],
    float64[::1],
    float64[::1],
    float64,
    int64,
    float64
), cache=True)
def gradient_descent(ds: NDArray[np.int64],
                     bias: float,
                     trend: float,
                     amps: NDArray[np.float64],
                     phases: NDArray[np.float64],
                     freqs: NDArray[np.float64],
                     regressors: NDArray[np.float64],
                     regressor_weights: NDArray[np.float64],
                     y: NDArray[np.float64],
                     learning_rate: float,
                     n_iterations: int,
                     tol: float
                     ):
    params = np.concatenate((np.array([bias, trend]), amps, phases, regressor_weights), axis=0)
    for _ in range(n_iterations):
        cost_derivative = find_cost_derivative(ds,
                                               bias,
                                               trend,
                                               amps,
                                               phases,
                                               freqs,
                                               regressors,
                                               regressor_weights,
                                               y
                                               )
        amps, phases, freqs = update_waves(amps, phases, freqs, ds, cost_derivative, learning_rate)
        bias = update_bias(bias, cost_derivative, learning_rate)
        trend = update_trend(trend, cost_derivative, ds, learning_rate)
        regressor_weights = update_regressor_weights(regressor_weights, regressors, cost_derivative, learning_rate)

        new_params = np.concatenate((np.array([bias, trend]), amps, phases, regressor_weights), axis=0)
        max_update = np.absolute(new_params - params).max()
        if max_update < tol:
            break
        params = np.copy(new_params)

    return (bias,
            trend,
            amps,
            phases,
            freqs,
            regressor_weights
            )
