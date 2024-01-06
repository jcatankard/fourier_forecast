from plotly.subplots import make_subplots
from numpy.typing import NDArray
from typing import Any, Optional
import plotly.graph_objs as go
import numpy as np


FIGSIZE = (900, 300)
COMPONENTS_TYPE = dict[str, dict[str, Any]]


def plot_seasonality_components(m, figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    components = get_seasonality_components(m)
    fig = create_fig(components, figsize)
    return fig


def plot_lag_components(m, figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    components = get_lag_components(m)
    fig = create_fig(components, figsize)
    return fig


def plot_regressor_components(m,
                              regressor_names: Optional[list[str]] = None,
                              figsize: tuple[int, int] = FIGSIZE
                              ) -> go.Figure:
    if regressor_names is None:
        regressor_names = [f"Regressor: {i + 1}" for i in range(m.n_regressors)]
    components = get_regressor_components(m, regressor_names)
    fig = create_fig(components, figsize)
    return fig


def plot_components(m, figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    trend = get_trend_component(m)
    seasonality = get_seasonality_components(m)
    all_lags = get_combined_lags_component(m)
    extra_regressors = get_extra_regressors_component(m)
    components = {
        **trend, **seasonality, **extra_regressors, **all_lags
    }
    fig = create_fig(components, figsize)
    return fig


def get_trend_component(m) -> COMPONENTS_TYPE:
    trend = m.x_[:, : m.seasonality_start_column] @ m.params_[: m.seasonality_start_column]
    trend = np.exp(trend) if m.log_y else trend
    range_y = [min(0, trend.min() * 1.25), max(0, trend.max() * 1.25)]
    name = 'trend'
    return {name: {
        'trace': get_trace(name, m.ds, trend),
        'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name), range=range_y)
    }
    }


def get_extra_regressors_component(m) -> COMPONENTS_TYPE:
    regs = m.x_[:, m.regressor_start_column:] @ m.params_[m.regressor_start_column:]
    name = 'extra_regressors'
    return {name: {
        'trace': get_trace(name, m.ds, regs),
        'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
    }
    }


def get_combined_lags_component(m) -> COMPONENTS_TYPE:
    lags = (m.x_[:, m.lag_start_column: m.regressor_start_column]
            @ m.params_[m.lag_start_column: m.regressor_start_column]
            )
    lags[: m.n_lags] = np.nan
    name = 'all_lags'
    return {name: {
        'trace': get_trace(name, m.ds, lags),
        'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
    }
    }


def get_seasonality_components(m) -> COMPONENTS_TYPE:
    start_col = m.seasonality_start_column
    components = {}
    for periods, n_terms in m.seasonality_terms.items():
        n_to_plot = np.ceil(periods).astype(np.int64)

        end_col = start_col + n_terms * 2
        y = m.x_[: n_to_plot, start_col: end_col] @ m.params_[start_col: end_col]
        name = f'seasonality: periods={periods}'

        components[name] = {
            'trace': get_trace(name, np.arange(n_to_plot), y),
            'xaxis': go.layout.XAxis(range=[0, n_to_plot - 1]),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
        }

        start_col = end_col

    return components


def get_regressor_components(m, regressor_names: list[str]) -> COMPONENTS_TYPE:
    components = {}
    for i, name in enumerate(regressor_names):
        col = m.regressor_start_column + i
        reg = m.x_[:, col] * m.params_[col]
        components[name] = {
            'trace': get_trace(name, m.ds, reg),
            'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
        }
    return components


def get_lag_components(m) -> COMPONENTS_TYPE:
    components = {}
    for i in range(m.n_lags):
        name = f"Lag term: {i + 1}"
        col = m.lag_start_column + i
        lag = m.x_[:, col] * m.params_[col]
        lag[: i + 1] = np.nan
        components[name] = {
            'trace': get_trace(name, m.ds, lag),
            'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
        }
    return components


def get_trace(name: str, x: NDArray, y: NDArray) -> go.Scatter:
    return go.Scatter(name=name, x=x, y=y, mode='lines', line=go.scatter.Line(width=2))


def create_fig(components: dict[str, Any], figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    fig = make_subplots(rows=len(components), cols=1, print_grid=False)
    fig['layout'].update(go.Layout(
        showlegend=False,
        width=figsize[0],
        height=figsize[1] * len(components)
    ))
    for i, name in enumerate(components):
        if i == 0:
            xaxis = fig['layout']['xaxis']
            yaxis = fig['layout']['yaxis']
        else:
            xaxis = fig['layout']['xaxis{}'.format(i + 1)]
            yaxis = fig['layout']['yaxis{}'.format(i + 1)]
        xaxis.update(components[name]['xaxis'])
        yaxis.update(components[name]['yaxis'])

        fig.add_trace(components[name]['trace'], i + 1, 1)
    return fig
