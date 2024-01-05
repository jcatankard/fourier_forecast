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


def create_data(regressors: bool,
                fourier_terms: list,
                log_y: bool = False
                ) -> tuple[NDArray[date], NDArray[np.float64], NDArray[np.float64]]:

    size = np.random.randint(366 * 2, 366 * 3)
    ds = np.arange(date.today() - timedelta(days=size), date.today()).astype('O')
    t = np.arange(size, dtype=np.int64)
    regressor_x = make_regressors(t, regressors)

    if log_y:
        y_clean = np.array([
            np.random.normal(0.05, .01) * t,  # trend
            np.random.randint(1, 100) * np.ones(size),  # bias
            np.sum(regressor_x, axis=1)  # regressors
        ]).sum(axis=0)
    else:
        y_clean = np.array([
            np.random.normal(0, .1) * t,  # trend
            np.random.choice([-1, 1]) * np.random.randint(1, 100) * np.ones(size),  # bias
            np.sum(regressor_x, axis=1)  # regressors
        ]).sum(axis=0)

    # weeks, months, quarters, years
    for i, periods in enumerate([7, 30.43, 91.31, 365.25]):
        for f in range(fourier_terms[i]):
            y_clean += make_wave(t, (f + 1) / periods)

    if log_y:
        y_clean = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min())
        y_clean = np.exp(y_clean + 10)

    return ds, y_clean, regressor_x
