from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def subplot(ax, ds: NDArray[np.int64], data: NDArray[np.float64], label: str):
    ax.plot(ds, data, label=label)
    ax.legend()


def plot_components(m):
    n_seasonalities = len(m.seasonality_terms)
    n_rows = 2 + np.ceil(n_seasonalities / 2).astype(np.int64)
    fig, ax = plt.subplots(nrows=n_rows, ncols=2)

    subplot(ax[0, 0], m.ds, m.x_[:, 0] * m.params_[0], 'bias')
    subplot(ax[0, 1], m.ds, m.x_[:, 1] * m.params_[1], 'trend')

    regs = m.x_[:, -m.n_regressors:] @ m.params_[-m.n_regressors:]
    subplot(ax[1, 0], m.ds, regs, 'regressors')

    subplot(ax[1, 1], m.ds, m.y - m.fitted(), 'noise')

    start_col = 2
    for i, (periods, n_terms) in enumerate(m.seasonality_terms.items()):
        n_to_plot = np.ceil(periods).astype(np.int64)
        ds = np.arange(n_to_plot)

        end_col = start_col + n_terms * 2
        s = m.x_[:, start_col: end_col] @ m.params_[start_col: end_col]

        row = 2 + i // 2
        col = i % 2
        subplot(ax[row, col], ds, s[: n_to_plot], f'seasonality: periods={periods}')

        start_col = end_col

    if n_seasonalities % 2 == 1:
        ax[n_rows - 1, 1].axis('off')
    plt.show()
