from numba import njit, int64, float64
from numpy.typing import NDArray
import numpy as np


@njit(float64[::1](int64[::1]), cache=True)
def x(ds: NDArray[np.int64]) -> NDArray[np.float64]:
    return 2 * np.pi * ds


@njit(float64[::1](float64, float64, float64, int64[::1]), cache=True)
def sin_wave(amplitude: float, phase: float, frequency: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    return amplitude * np.sin(phase + x(ds) * frequency)


@njit(float64[::1](float64, float64, float64, int64[::1]), cache=True)
def cos_wave(amplitude: float, phase: float, frequency: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    return amplitude * np.cos(phase + x(ds) * frequency)


@njit(float64[::1](float64, float64, int64[::1]), cache=True)
def dy_da(phase: float, frequency: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    """differentiate y with respect to amplitude"""
    return sin_wave(1., phase, frequency, ds)


@njit(float64[::1](float64, float64, float64, int64[::1]), cache=True)
def dy_dp(amplitude: float, phase: float, frequency: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    """differentiate y with respect to phase"""
    return cos_wave(amplitude, phase, frequency, ds)


@njit(float64[::1](float64, float64, float64, int64[::1]), cache=True)
def dy_df(amplitude: float, phase: float, frequency: float, ds: NDArray[np.int64]) -> NDArray[np.float64]:
    """differentiate y with respect to frequency"""
    return x(ds) * cos_wave(amplitude, phase, frequency, ds)
