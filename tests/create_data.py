from datetime import date, timedelta
from numpy.typing import NDArray
import numpy as np


def make_wave(t: NDArray[np.int64], freq: float) -> NDArray[np.float64]:
    a = np.random.rand() * 10
    p = np.random.normal(0, 2)
    return a * np.sin(p + 2 * np.pi * freq * t)


def make_regressors(t: NDArray[np.int64], regressors: bool) -> NDArray:
    p = np.random.normal(.5, .1) if regressors else 1.
    n_features = np.random.randint(2, 20) if regressors else 0
    return np.random.choice([0, 1], size=t.size * n_features, p=[p, 1 - p]).reshape(t.size, -1)


def make_trend(t: NDArray[np.int64], growth: str, log_y: bool) -> NDArray:
    linear = t.astype(np.float64) / t.size
    if growth == 'linear':
        trend = linear
    elif growth == 'logistic':
        trend = 1 / (1 + np.exp((0.5 - linear) * 10))
    elif growth == 'logarithmic':
        trend = np.log(linear * (np.e - 1) + 1)
    else:
        trend = np.zeros_like(t, dtype=np.float64)

    direction = np.random.normal(0.05, .01) if log_y else np.random.normal(0, .1)
    return direction * trend * t.size


def make_bias(t: NDArray[np.int64], log_y: bool) -> NDArray:
    sign = 1 if log_y else np.random.choice([-1, 1])
    return sign * np.random.randint(1, 100) * np.ones(t.size)


def create_data(regressors: bool,
                fourier_terms: list,
                log_y: bool = False,
                growth: str = 'linear'
                ) -> tuple[NDArray[date], NDArray[np.float64], NDArray[np.float64]]:

    size = np.random.randint(366 * 2, 366 * 3)
    ds = np.arange(date.today() - timedelta(days=size), date.today()).astype('O')
    t = np.arange(size, dtype=np.int64)
    regressor_x = make_regressors(t, regressors)

    y_clean = np.array([
        make_trend(t, growth, log_y),
        make_bias(t, log_y),
        np.sum(regressor_x, axis=1)
    ]).sum(axis=0)

    # weeks, months, quarters, years
    for i, periods in enumerate([7, 30.43, 91.31, 365.25]):
        for f in range(fourier_terms[i]):
            y_clean += make_wave(t, (f + 1) / periods)

    if log_y:
        y_clean = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min())
        y_clean = np.exp(y_clean + 10)

    return ds, y_clean, regressor_x
